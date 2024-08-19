import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END

from nodes.execute_step import ExecuteStepState, execute_step


# Mock tool for testing
def mock_search(query: str) -> str:
    return f"Mocked search results for: {query}"


search_tool = StructuredTool.from_function(
    func=mock_search,
    name="search",
    description="Search the web for information."
)


@pytest.fixture
def tools():
    return [search_tool]


@pytest.mark.asyncio
async def test_execute_step_with_valid_tool_call(tools):
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

    result = await execute_step(initial_state, tools)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert "Mocked search results for: LangGraph usage examples" in result["messages"][0].content


@pytest.mark.asyncio
async def test_execute_step_with_no_tool_calls():
    initial_state = ExecuteStepState(
        messages=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
    )

    result = await execute_step(initial_state, [])

    assert result["messages"] == []


@pytest.mark.asyncio
async def test_execute_step_with_invalid_tool():
    initial_state = ExecuteStepState(
        messages=[
            AIMessage(content="", tool_calls=[{
                "id": "call_1",
                "name": "invalid_tool",
                "arguments": '{}'
            }])
        ]
    )

    result = await execute_step(initial_state, [])

    assert len(result["messages"]) == 1
    assert "Error executing tools:" in result["messages"][0].content


@pytest.mark.asyncio
async def test_execute_step_in_graph():
    workflow = StateGraph(ExecuteStepState)

    workflow.add_node("execute_step", lambda state: execute_step(state, tools=[search_tool]))
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

    assert len(result["messages"]) == 3  # Initial 2 messages + 1 new ToolMessage
    assert isinstance(result["messages"][-1], ToolMessage)
    assert "Mocked search results for: LangGraph usage examples" in result["messages"][-1].content


if __name__ == "__main__":
    pytest.main([__file__])
