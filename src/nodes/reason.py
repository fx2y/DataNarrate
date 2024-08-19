from typing import Dict, Any, List, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


class ReasoningOutput(BaseModel):
    """Structured output for reasoning step"""
    analysis: str = Field(description="Analysis of the current state and results")
    next_action: Literal["continue", "revise", "finish"] = Field(description="Recommended next action")
    explanation: str = Field(description="Explanation for the recommended next action")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="List of tool calls to make")


class ReasoningConfig(BaseModel):
    """Configuration for the reasoning node"""
    llm: BaseChatModel = Field(default_factory=lambda: ChatAnthropic(model="claude-3-haiku-20240307"))
    prompt: ChatPromptTemplate = Field(
        default_factory=lambda: ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert reasoning engine. Analyze the current state and results, then recommend the next action."),
            ("human",
             "Current state: {state}\nLast action result: {last_result}\n\nAnalyze the situation and recommend the next action. If needed, use available tools."),
        ])
    )


class ReasoningState(MessagesState):
    """State for the reasoning node"""
    context: Dict[str, Any] = Field(default_factory=dict)
    reasoning_output: ReasoningOutput = Field(default=None)


class ReasoningNode:
    """A node for performing reasoning in a LangGraph"""

    def __init__(self, config: ReasoningConfig = ReasoningConfig()):
        """Initialize the reasoning node with the given configuration"""
        self.config = config
        self.reasoning_chain = self.config.prompt | self.config.llm.with_structured_output(ReasoningOutput)

    async def __call__(self, state: ReasoningState) -> Dict[str, Any]:
        """
        Perform reasoning on the current state and decide the next action.

        Args:
            state (ReasoningState): The current state of the graph

        Returns:
            Dict[str, Any]: Updated state with reasoning output
        """
        try:
            result = await self.reasoning_chain.ainvoke({
                "state": state.context,
                "last_result": state.messages[-1].content if state.messages else "No previous actions"
            })

            new_message = AIMessage(
                content=f"Reasoning: {result.analysis}\nNext action: {result.next_action}\nExplanation: {result.explanation}")

            return {
                "messages": [new_message],
                "reasoning_output": result
            }
        except Exception as e:
            error_message = f"Error in reasoning: {str(e)}"
            return {
                "messages": [AIMessage(content=error_message)],
                "reasoning_output": ReasoningOutput(
                    analysis="Error occurred during reasoning",
                    next_action="revise",
                    explanation=error_message
                )
            }


def create_reasoning_node(config: ReasoningConfig = ReasoningConfig()) -> ToolNode:
    """Create a ToolNode for reasoning"""
    return ToolNode(ReasoningNode(config))


async def add_reasoning_to_graph(graph: StateGraph, config: ReasoningConfig = ReasoningConfig()):
    """Add the reasoning node to the graph"""
    reasoning_node = create_reasoning_node(config)
    graph.add_node("reason", reasoning_node)

    graph.add_conditional_edges(
        "reason",
        lambda x: x["reasoning_output"].next_action,
        {
            "continue": "execute_step",
            "revise": "plan_task",
            "finish": "generate_output"
        }
    )

    # Add edge for tool execution if tool calls are present
    graph.add_conditional_edges(
        "reason",
        lambda x: "execute_tools" if x["reasoning_output"].tool_calls else x["reasoning_output"].next_action,
        {
            "execute_tools": "execute_tools"
        }
    )


# For testing and demonstration purposes
async def test_reasoning_node():
    config = ReasoningConfig()
    node = ReasoningNode(config)
    test_state = ReasoningState(
        context={"task": "Analyze data"},
        messages=[HumanMessage(content="Please analyze the sales data")]
    )
    result = await node(test_state)
    assert "messages" in result
    assert "reasoning_output" in result
    print("Test passed successfully")
    print(f"Reasoning output: {result['reasoning_output']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_reasoning_node())
