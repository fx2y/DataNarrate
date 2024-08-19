# nodes/plan_task.py

from typing import List, Dict, Any, Optional

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field


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
        state: Dict[str, Any],
        llm: BaseChatModel,
        tools: List[BaseTool],
        callbacks: Optional[CallbackManagerForChainRun] = None
) -> Dict[str, Any]:
    """
    Generate a plan and select tools based on the input in the state.
    """
    try:
        tool_descriptions = get_tool_descriptions(tools)
        prompt_args = {"input": state.get("input", ""), "tool_descriptions": tool_descriptions}
        llm_output = llm.predict_messages([planner_prompt.format_messages(**prompt_args)[0]], callbacks=callbacks)

        plan = parse_llm_output(llm_output.content)

        return {
            **state,
            "plan": plan.dict(),
            "current_step": 0,
            "status": "planning_complete"
        }
    except Exception as e:
        if callbacks:
            callbacks.on_chain_error(e, verbose=True)
        return {
            **state,
            "error": f"Planning and tool selection failed: {str(e)}",
            "status": "planning_failed"
        }


# Graph construction
def create_planning_graph(llm: BaseChatModel, tools: List[BaseTool]) -> StateGraph:
    workflow = StateGraph()

    workflow.add_node("plan_and_select", lambda state: plan_task_and_select_tools(state, llm, tools))

    workflow.set_entry_point("plan_and_select")

    workflow.add_conditional_edges(
        "plan_and_select",
        lambda x: END if x["status"] == "planning_complete" else "plan_and_select"
    )

    return workflow.compile()

# Example usage:
# from langchain_openai import ChatOpenAI
# from langchain_community.tools import DuckDuckGoSearchRun
#
# llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
# tools = [DuckDuckGoSearchRun()]
# planning_graph = create_planning_graph(llm, tools)
# result = planning_graph.invoke({"input": "What's the weather like in San Francisco?"})
