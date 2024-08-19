from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from nodes.plan_task import (
    get_tool_descriptions,
    parse_llm_output,
    plan_task_and_select_tools,
    create_planning_graph,
    Plan,
)


class MockTool(BaseTool):
    name: str
    description: str

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return f"Mock result for {self.name}"


class MockLLM(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return AIMessage(content="""
Step 1: Check the current weather in San Francisco | WeatherTool
Step 2: Analyze the weather data | DataAnalysisTool
Final Answer: The weather in San Francisco is sunny with a temperature of 72°F.
""")


@pytest.fixture
def mock_tools():
    return [
        MockTool(name="WeatherTool", description="Get current weather information"),
        MockTool(name="DataAnalysisTool", description="Analyze data and provide insights"),
    ]


@pytest.fixture
def mock_llm():
    return MockLLM()


def test_get_tool_descriptions(mock_tools):
    descriptions = get_tool_descriptions(mock_tools)
    assert "WeatherTool: Get current weather information" in descriptions
    assert "DataAnalysisTool: Analyze data and provide insights" in descriptions


def test_parse_llm_output():
    llm_output = """
Step 1: Check the current weather in San Francisco | WeatherTool
Step 2: Analyze the weather data | DataAnalysisTool
Final Answer: The weather in San Francisco is sunny with a temperature of 72°F.
"""
    plan = parse_llm_output(llm_output)
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 2
    assert plan.steps[0].description == "Check the current weather in San Francisco"
    assert plan.steps[0].tools[0].tool == "WeatherTool"
    assert plan.final_answer == "The weather in San Francisco is sunny with a temperature of 72°F."


def test_plan_task_and_select_tools(mock_llm, mock_tools):
    state = {"input": "What's the weather like in San Francisco?"}
    result = plan_task_and_select_tools(state, mock_llm, mock_tools)

    assert result["status"] == "planning_complete"
    assert len(result["plan"]["steps"]) == 2
    assert result["current_step"] == 0


def test_create_planning_graph(mock_llm, mock_tools):
    planning_graph = create_planning_graph(mock_llm, mock_tools)
    result = planning_graph.invoke({"input": "What's the weather like in San Francisco?"})

    assert result["status"] == "planning_complete"
    assert len(result["plan"]["steps"]) == 2
    assert result["current_step"] == 0


def test_error_handling(mock_tools):
    class ErrorLLM(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise Exception("LLM error")

    state = {"input": "What's the weather like in San Francisco?"}
    result = plan_task_and_select_tools(state, ErrorLLM(), mock_tools)

    assert result["status"] == "planning_failed"
    assert "Planning and tool selection failed" in result["error"]


# Integration test
def test_end_to_end_workflow(mock_llm, mock_tools):
    planning_graph = create_planning_graph(mock_llm, mock_tools)
    initial_state = {"input": "What's the weather like in San Francisco?"}

    result = planning_graph.invoke(initial_state)

    assert result["status"] == "planning_complete"
    assert len(result["plan"]["steps"]) == 2
    assert result["plan"]["steps"][0]["description"] == "Check the current weather in San Francisco"
    assert result["plan"]["steps"][0]["tools"][0]["tool"] == "WeatherTool"
    assert result["plan"]["steps"][1]["description"] == "Analyze the weather data"
    assert result["plan"]["steps"][1]["tools"][0]["tool"] == "DataAnalysisTool"
    assert "The weather in San Francisco is sunny" in result["plan"]["final_answer"]
