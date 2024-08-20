import getpass
import os
import re
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Sequence, List, Dict, Any, Union, Iterable

from langchain import hub
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBranch, as_runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from output_parser import LLMCompilerPlanParser, Task
from typing_extensions import TypedDict


# Helper function to get environment variables
def _get_pass(var: str):
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"{var}: ")


# Set up environment variables for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LLMCompiler"
_get_pass("LANGCHAIN_API_KEY")
_get_pass("OPENAI_API_KEY")
_get_pass("TAVILY_API_KEY")

# Define tools
calculate = get_math_tool(ChatOpenAI(model="gpt-4-turbo-preview"))
search = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="the search query") - a search engine.',
)
tools = [search, calculate]


# Planner creation function
def create_planner(llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate):
    tool_descriptions = "\n".join(
        f"{i + 1}. {tool.description}\n" for i, tool in enumerate(tools)
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )
    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
               "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
               'You MUST use these information to create the next plan under "Current Plan".\n'
               ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
               " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
               " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list):
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = max(next_task, int(message.name.split("_")[-1]) + 1)
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return (
            RunnableBranch(
                (should_replan, wrap_and_get_last_index | replanner_prompt),
                wrap_messages | planner_prompt,
            )
            | llm
            | LLMCompilerPlanParser(tools=tools)
    )


# Task Fetching Unit
def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            idx = int(message.name.split("_")[-1])
            results[idx] = message.content
    return results


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {k: _resolve_arg(v, observations) for k, v in args.items()}
        else:
            resolved_args = args
    except Exception as e:
        return f"ERROR(Failed to call {tool_to_use.name} with args {args}. Args could not be resolved. Error: {repr(e)})"
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return f"ERROR(Failed to call {tool_to_use.name} with args {args}. Args resolved to {resolved_args}. Error: {repr(e)})"


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
def schedule_task(task_inputs, config):
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        observation = f"ERROR(Failed to execute task {task['idx']})"
    observations[task["idx"]] = observation


def schedule_pending_task(task: Task, observations: Dict[int, Any], retry_after: float = 0.2):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    tasks = scheduler_input["tasks"]
    messages = scheduler_input["messages"]
    observations = _get_observations(messages)
    futures = []
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            futures.append(executor.submit(schedule_pending_task, task, observations))
        wait(futures)
    new_observations = {
        k: FunctionMessage(name=f"task_{k}", content=v) for k, v in observations.items()
    }
    return list(new_observations.values())


# Joiner
class FinalResponse(BaseModel):
    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed.")


class JoinOutputs(BaseModel):
    thought: str = Field(description="The chain of thought reasoning for the selected action")
    action: Union[FinalResponse, Replan]


joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(examples="")
llm = ChatOpenAI(model="gpt-4-turbo-preview")
runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)


def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        response.append(SystemMessage(content=decision.action.feedback))
    else:
        response.append(AIMessage(content=decision.action.response))
    return response


def select_recent_messages(state) -> dict:
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        if isinstance(msg, HumanMessage) or isinstance(msg, FunctionMessage):
            selected.append(msg)
        if len(selected) >= 10:
            break
    return {"messages": selected[::-1]}


joiner = select_recent_messages | runnable | _parse_joiner_output


# Compose using LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)
graph_builder.add_edge("plan_and_schedule", "join")


def should_continue(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage):
        return "plan_and_schedule"
    return END


graph_builder.add_conditional_edges(start_key="join", condition=should_continue)
graph_builder.add_edge(START, "plan_and_schedule")
chain = graph_builder.compile()

# Example usage
example_question = "What's the temperature in SF raised to the 3rd power?"
for step in chain.stream({"messages": [HumanMessage(content=example_question)]}):
    print(step)
    print("---")
