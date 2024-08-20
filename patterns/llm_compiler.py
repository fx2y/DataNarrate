import getpass
import itertools
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Sequence, List, Dict, Any, Union, Iterable, Annotated

from langchain import hub
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBranch, chain as as_runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from math_tools import get_math_tool
from output_parser import LLMCompilerPlanParser, Task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper function to get environment variables
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


# Set up environment variables for tracing
def setup_environment():
    os.environ["LANGCHAIN_TRACING_V2"] = "True"
    os.environ["LANGCHAIN_PROJECT"] = "LLMCompiler"
    _set_if_undefined("LANGCHAIN_API_KEY")
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")


setup_environment()

# Define tools
calculate = get_math_tool(ChatOpenAI(model="gpt-4-turbo-preview"))
search = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="the search query") - a search engine.',
)
tools = [search, calculate]


# Planner creation function
def create_planner(llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate):
    """Create a planner for the given LLM and tools.

    Args:
        llm (BaseChatModel): The language model to use.
        tools (Sequence[BaseTool]): The tools available to the planner.
        base_prompt (ChatPromptTemplate): The base prompt template.

    Returns:
        RunnableBranch: The planner runnable branch.
    """
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
    """Extract observations from messages.

    Args:
        messages (List[BaseMessage]): The list of messages.

    Returns:
        Dict[int, Any]: A dictionary of observations indexed by task ID.
    """
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
    """Execute a given task with resolved arguments.

    Args:
        task (Task): The task to execute.
        observations (Dict[int, Any]): The observations from previous tasks.
        config (Any): The configuration for the task.

    Returns:
        Any: The result of the task execution.
    """
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
        logger.error(f"Failed to resolve args for {tool_to_use.name} with args {args}. Error: {repr(e)}")
        return f"ERROR(Failed to call {tool_to_use.name} with args {args}. Args could not be resolved. Error: {repr(e)})"
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        logger.error(f"Failed to call {tool_to_use.name} with resolved args {resolved_args}. Error: {repr(e)}")
        return f"ERROR(Failed to call {tool_to_use.name} with args {args}. Args resolved to {resolved_args}. Error: {repr(e)})"


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    """Resolve arguments using observations.

    Args:
        arg (Union[str, Any]): The argument to resolve.
        observations (Dict[int, Any]): The observations from previous tasks.

    Returns:
        Any: The resolved argument.
    """
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
    """Schedule a single task for execution.

    Args:
        task_inputs (Dict[str, Any]): The inputs for the task.
        config (Any): The configuration for the task.
    """
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        observation = f"ERROR(Failed to execute task {task['idx']})"
    observations[task["idx"]] = observation


def schedule_pending_task(task: Task, observations: Dict[int, Any], retry_after: float = 0.2):
    """Schedule a pending task for execution.

    Args:
        task (Task): The task to schedule.
        observations (Dict[int, Any]): The observations from previous tasks.
        retry_after (float, optional): The retry interval. Defaults to 0.2.
    """
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Schedule multiple tasks for execution.

    Args:
        scheduler_input (SchedulerInput): The input for the scheduler.

    Returns:
        List[FunctionMessage]: The list of function messages.
    """
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


llm = ChatOpenAI(model="gpt-4-turbo-preview")
prompt = hub.pull("wfh/llm-compiler")
planner = create_planner(llm, tools, prompt)


@as_runnable
def plan_and_schedule(state):
    messages = state["messages"]
    tasks = planner.stream(messages)
    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # Handle the case where tasks is empty.
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        }
    )
    return {"messages": [scheduled_tasks]}


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
runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)


def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    """Parse the output from the joiner.

    Args:
        decision (JoinOutputs): The decision from the joiner.

    Returns:
        List[BaseMessage]: The list of messages.
    """
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        response.append(SystemMessage(content=decision.action.feedback))
    else:
        response.append(AIMessage(content=decision.action.response))
    return response


def select_recent_messages(state) -> dict:
    """Select recent messages from the state.

    Args:
        state (dict): The state containing messages.

    Returns:
        dict: The selected recent messages.
    """
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
    """Determine if the process should continue.

    Args:
        state (dict): The current state.

    Returns:
        str: The next step in the process.
    """
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
