from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .config import settings
from .nodes import (
    analyze_query,
    plan_task,
    select_tool,
    execute_step,
    reason,
    generate_output,
    human_node
)
from .state import DataNarrationState
from .tools import get_tools


def create_data_narration_graph():
    workflow = StateGraph(DataNarrationState)

    # Add nodes
    nodes = {
        "analyze_query": analyze_query,
        "plan_task": plan_task,
        "select_tool": select_tool,
        "execute_step": execute_step,
        "reason": reason,
        "generate_output": generate_output,
        "human": human_node,
        "tools": ToolNode(get_tools())
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # Add edges
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "plan_task")
    workflow.add_edge("plan_task", "select_tool")
    workflow.add_edge("select_tool", "execute_step")
    workflow.add_edge("execute_step", "reason")

    def select_next_node(state: DataNarrationState) -> Literal["human", "tools", "select_tool", "__end__"]:
        if state.error:
            return "__end__"
        if settings.ENABLE_HUMAN_FEEDBACK and state.dialog_state and state.dialog_state[-1] == "ask_human":
            return "human"
        if state.current_step < len(state.task_plan) and state.current_step < settings.MAX_ITERATIONS:
            return "select_tool"
        return "__end__"

    workflow.add_conditional_edges(
        "reason",
        select_next_node,
        {
            "human": "human",
            "tools": "tools",
            "select_tool": "select_tool",
            "__end__": "generate_output",
        }
    )
    workflow.add_edge("human", "reason")
    workflow.add_edge("tools", "reason")
    workflow.add_edge("generate_output", END)

    return workflow.compile()
