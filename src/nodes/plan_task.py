from typing import List, Dict, Any, Annotated, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


# Define models
class ToolSelection(BaseModel):
    tool: str
    args: Dict[str, Union[str, int, float, bool, Dict, List]] = Field(default_factory=dict)


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
    query_analysis: Dict[str, Any] = Field(default_factory=dict)


# Prompts
SYSTEM_PROMPT = """For the given objective, create a step-by-step plan and select appropriate tools for each step.
Available tools:
{tool_descriptions}

Query Analysis:
{query_analysis}

Guidelines:
1. Each step should be concise and actionable.
2. Select one or more tools for each step if needed. If no tool is required, leave it empty.
3. For tool arguments, you can use:
   - Placeholders like $0, $1, $2, etc., to refer to outputs from specific previous steps.
   - Direct values for primitives, dicts, or lists.
4. When using database tools, refer to the required_data_sources from the query analysis to construct more accurate queries.
5. The final step should provide the answer to the objective.

Output Format:
Step 1: [Description] | [Tool1(arg1=$0, arg2="literal value"), Tool2(arg1={{"key": "value"}})]
Step 2: [Description] | [Tool1(arg1=$1, arg2=[1, 2, 3])]
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
            tools = []
            if len(parts) > 1:
                tool_strings = parts[1].strip().split("),")
                for tool_string in tool_strings:
                    tool_parts = tool_string.split("(")
                    tool_name = tool_parts[0].strip()
                    args = {}
                    if len(tool_parts) > 1:
                        arg_strings = tool_parts[1].rstrip(")").split(",")
                        for arg_string in arg_strings:
                            key, value = arg_string.split("=")
                            args[key.strip()] = eval(value.strip())  # Safely evaluate the argument value
                    tools.append(ToolSelection(tool=tool_name, args=args))
            steps.append(PlanStep(description=description, tools=tools))
        elif line.startswith("Final Answer:"):
            final_answer = line.split(":", 1)[1].strip()

    return Plan(steps=steps, final_answer=final_answer)


async def plan_task_and_select_tools(
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
        prompt_args = {
            "input": input_message,
            "tool_descriptions": tool_descriptions,
            "query_analysis": state.query_analysis
        }
        llm_output = await llm.ainvoke([planner_prompt.format_messages(**prompt_args)[0]])

        plan = parse_llm_output(llm_output.content)

        return PlanningState(
            messages=state.messages + [AIMessage(content=llm_output.content)],
            plan=plan,
            current_step=0,
            status="planning_complete",
            query_analysis=state.query_analysis
        )
    except Exception as e:
        return PlanningState(
            messages=state.messages + [AIMessage(content=f"Planning and tool selection failed: {str(e)}")],
            status="planning_failed",
            query_analysis=state.query_analysis
        )


# Graph construction
def create_planning_graph(llm: BaseChatModel, tools: List[BaseTool]) -> StateGraph:
    workflow = StateGraph(PlanningState)

    workflow.add_node("plan_and_select", lambda state: plan_task_and_select_tools(state, llm, tools))

    workflow.set_entry_point("plan_and_select")

    return workflow
