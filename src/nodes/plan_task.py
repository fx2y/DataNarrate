from typing import List, Dict, Any, Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


# Define models
class ToolSelection(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    description: str
    tools: List[ToolSelection] = Field(default_factory=list)


class Plan(BaseModel):
    steps: List[PlanStep]
    final_answer: str


# Define state
class PlanningState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    plan: Plan = None
    current_step: int = 0
    status: str = "planning"


# Prompts
SYSTEM_PROMPT = """For the given objective, create a step-by-step plan and select appropriate tools for each step.
Available tools:
{tool_descriptions}

Guidelines:
1. Each step should be concise and actionable.
2. Select one or more tools for each step if needed. If no tool is required, leave it empty.
3. The final step should provide the answer to the objective.

Output Format:
Step 1: [Description] | [Tool1, Tool2, ...]
Step 2: [Description] | [Tool1, ...]
...
Final Answer: [Description of the final answer]
"""

HUMAN_PROMPT = "Objective: {input}"

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT)
])


def get_tool_descriptions(tools: List[BaseTool]) -> str:
    return "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])


def parse_llm_output(output: str) -> Plan:
    lines = output.strip().split('\n')
    steps = []
    final_answer = ""

    for line in lines:
        if line.startswith("Step"):
            parts = line.split("|")
            description = parts[0].split(":", 1)[1].strip()
            tools = [ToolSelection(tool=tool.strip()) for tool in parts[1].strip().split(",")] if len(parts) > 1 else []
            steps.append(PlanStep(description=description, tools=tools))
        elif line.startswith("Final Answer:"):
            final_answer = line.split(":", 1)[1].strip()

    return Plan(steps=steps, final_answer=final_answer)


def plan_task_and_select_tools(
        state: PlanningState,
        llm: BaseChatModel,
        tools: List[BaseTool],
) -> PlanningState:
    """
    Generate a plan and select tools based on the input in the state.
    """
    try:
        tool_descriptions = get_tool_descriptions(tools)
        input_message = state.messages[-1].content if state.messages else ""
        prompt_args = {"input": input_message, "tool_descriptions": tool_descriptions}
        llm_output = llm.predict_messages([planner_prompt.format_messages(**prompt_args)[0]])

        plan = parse_llm_output(llm_output.content)

        return PlanningState(
            messages=state.messages + [AIMessage(content=llm_output.content)],
            plan=plan,
            current_step=0,
            status="planning_complete"
        )
    except Exception as e:
        return PlanningState(
            messages=state.messages + [AIMessage(content=f"Planning and tool selection failed: {str(e)}")],
            status="planning_failed"
        )


def should_continue(state: PlanningState) -> str:
    if state.status == "planning_complete":
        return END
    elif state.status == "planning_failed":
        return "plan_and_select"
    else:
        return "plan_and_select"


# Graph construction
def create_planning_graph(llm: BaseChatModel, tools: List[BaseTool]) -> CompiledStateGraph:
    workflow = StateGraph(PlanningState)

    workflow.add_node("plan_and_select", lambda state: plan_task_and_select_tools(state, llm, tools))

    workflow.set_entry_point("plan_and_select")

    workflow.add_conditional_edges(
        "plan_and_select",
        should_continue
    )

    return workflow.compile()

# Example usage:
# from langchain_openai import ChatOpenAI
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_core.messages import HumanMessage
#
# llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
# tools = [DuckDuckGoSearchRun()]
# planning_graph = create_planning_graph(llm, tools)
# initial_state = PlanningState(messages=[HumanMessage(content="What's the weather like in San Francisco?")])
# result = planning_graph.invoke(initial_state)
