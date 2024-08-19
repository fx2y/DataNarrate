from operator import add
from typing import Annotated, Dict, List

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode


class ExecuteStepState(Dict):
    messages: Annotated[List[BaseMessage], add]


async def execute_step(state: ExecuteStepState, tools: List[BaseTool]) -> Dict:
    """
    Execute a single step in the task plan using the selected tools.

    This function processes the last message's tool calls, executes the appropriate tools,
    and returns the results as messages to be added to the state.
    """
    tool_node = ToolNode(tools)

    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {"messages": []}

    try:
        result = await tool_node.ainvoke({"messages": [last_message]})
        return {"messages": result["messages"]}
    except Exception as e:
        error_message = f"Error executing tools: {str(e)}"
        return {"messages": [BaseMessage(content=error_message)]}


# Example usage in a LangGraph context
if __name__ == "__main__":
    from langgraph.graph import StateGraph, END
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage


    @tool
    def search(query: str) -> str:
        """Search the web for information."""
        return f"Search results for: {query}"


    workflow = StateGraph(ExecuteStepState)

    workflow.add_node("execute_step", lambda state: execute_step(state, tools=[search]))

    workflow.add_edge("execute_step", END)

    app = workflow.compile()

    initial_state = ExecuteStepState(
        messages=[
            HumanMessage(content="Find information about LangGraph"),
            AIMessage(content="", tool_calls=[{
                "id": "call_1",
                "name": "search",
                "arguments": '{"query": "LangGraph usage examples"}'
            }])
        ]
    )

    result = app.invoke(initial_state)
    print(result["messages"])
