from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from your_module_path import AnalyzeQueryNode, QueryAnalysis


@pytest.fixture
def analyze_query_node():
    return AnalyzeQueryNode()


def test_analyze_query_node_initialization(analyze_query_node):
    assert analyze_query_node.model is not None
    assert analyze_query_node.prompt is not None
    assert analyze_query_node.output_parser is not None


@pytest.mark.asyncio
async def test_analyze_query_node_run(analyze_query_node):
    mock_state = {
        "messages": [
            HumanMessage(content="Analyze the sales data for Q1 and visualize the trends.")
        ]
    }

    with patch.object(analyze_query_node, '_analyze_query') as mock_analyze:
        mock_analyze.return_value = QueryAnalysis(
            task_type="data analysis and visualization",
            sub_tasks=["Analyze Q1 sales data", "Visualize sales trends"],
            required_data_sources=["Q1 sales data"],
            constraints=[],
            potential_insights=["Sales trends over Q1"]
        )

        result = await analyze_query_node.run(mock_state)

        assert "query_analysis" in result
        assert isinstance(result["messages"][-1], AIMessage)
        assert "task_type" in result["query_analysis"]
        assert result["query_analysis"]["task_type"] == "data analysis and visualization"


@pytest.mark.asyncio
async def test_analyze_query_node_run_no_user_message(analyze_query_node):
    mock_state = {"messages": []}

    result = await analyze_query_node.run(mock_state)

    assert result == mock_state


@pytest.mark.asyncio
async def test_analyze_query_node_run_error_handling(analyze_query_node):
    mock_state = {
        "messages": [
            HumanMessage(content="Analyze the sales data for Q1 and visualize the trends.")
        ]
    }

    with patch.object(analyze_query_node, '_analyze_query', side_effect=Exception("Test error")):
        result = await analyze_query_node.run(mock_state)

        assert "error" in result
        assert result["error"].startswith("Failed to analyze query")


## Integration Test

import pytest


@pytest.mark.asyncio
async def test_analyze_query_integration():
    initial_state = {
        "messages": [
            HumanMessage(content="Analyze the sales data for Q1 and visualize the trends.")
        ]
    }

    result = await analyze_query(initial_state)

    assert "query_analysis" in result
    assert isinstance(result["query_analysis"], dict)
    assert "task_type" in result["query_analysis"]
    assert len(result["messages"]) == 2  # Original message + AI response


## Demo Script

import asyncio
from your_module_path import analyze_query
from langchain_core.messages import HumanMessage


async def main():
    initial_state = {
        "messages": [
            HumanMessage(content="Analyze the sales data for Q1 and visualize the trends.")
        ]
    }

    print("Initial State:")
    print(initial_state)

    result = await analyze_query(initial_state)

    print("\nFinal State:")
    print(result)

    print("\nQuery Analysis:")
    print(result["query_analysis"])


if __name__ == "__main__":
    asyncio.run(main())
