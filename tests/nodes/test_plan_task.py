# tests/test_plan_task.py

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from nodes.plan_task import create_planner


class MockChatModel(BaseChatModel):
    def __init__(self, response):
        self.response = response

    def invoke(self, messages, **kwargs):
        return AIMessage(content=self.response)

    def generate(self, messages, **kwargs):
        return ChatGeneration(message=AIMessage(content=self.response))


def test_create_planner():
    mock_llm = MockChatModel('{"steps": ["Step 1", "Step 2", "Step 3"]}')
    planner = create_planner(mock_llm)
    result = planner.invoke({"input": "Test input"})
    assert result["plan"] == ["Step 1", "Step 2", "Step 3"]
    assert result["current_step"] == 0
    assert result["status"] == "planning_complete"


def test_plan_task_success():
    mock_llm = MockChatModel('{"steps": ["Research", "Analyze", "Conclude"]}')
    state = {"input": "Investigate climate change"}
    result = plan_task(state, mock_llm)
    assert result["plan"] == ["Research", "Analyze", "Conclude"]
    assert result["current_step"] == 0
    assert result["status"] == "planning_complete"
    assert "error" not in result


def test_plan_task_failure():
    def failing_llm(*args, **kwargs):
        raise Exception("LLM failed")

    mock_llm = MockChatModel("Irrelevant response")
    mock_llm.invoke = failing_llm

    state = {"input": "Investigate climate change"}
    result = plan_task(state, mock_llm)
    assert "error" in result
    assert result["status"] == "planning_failed"


def test_plan_task_preserves_state():
    mock_llm = MockChatModel('{"steps": ["Step 1", "Step 2"]}')
    state = {"input": "Test input", "preserve_me": "important data"}
    result = plan_task(state, mock_llm)
    assert "preserve_me" in result
    assert result["preserve_me"] == "important data"


## Integration Test

# tests/test_plan_task_integration.py

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from nodes.plan_task import plan_task


def test_plan_task_in_graph():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    workflow = StateGraph()

    workflow.add_node("plan", lambda state: plan_task(state, llm))
    workflow.add_node("end", lambda state: {"result": "Plan created", **state})

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "end")
    workflow.add_edge("end", END)

    graph = workflow.compile()

    result = graph.invoke({"input": "Create a marketing strategy for a new smartphone"})

    assert "plan" in result
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) > 0
    assert result["status"] == "planning_complete"
    assert result["result"] == "Plan created"


if __name__ == "__main__":
    test_plan_task_in_graph()
