from typing import Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode


class ReasoningOutput(BaseModel):
    """Structured output for reasoning step"""
    analysis: str = Field(description="Analysis of the current state and results")
    next_action: str = Field(description="Recommended next action: 'continue', 'revise', or 'finish'")
    explanation: str = Field(description="Explanation for the recommended next action")


class ReasoningConfig(BaseModel):
    """Configuration for the reasoning node"""
    llm: BaseChatModel = Field(default_factory=lambda: ChatAnthropic(model="claude-3-haiku-20240307"))
    prompt: ChatPromptTemplate = Field(
        default_factory=lambda: ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert reasoning engine. Analyze the current state and results, then recommend the next action."),
            ("human",
             "Current state: {state}\nLast action result: {last_result}\n\nAnalyze the situation and recommend the next action."),
        ])
    )


class ReasoningNode:
    """A node for performing reasoning in a LangGraph"""

    def __init__(self, config: ReasoningConfig = ReasoningConfig()):
        """Initialize the reasoning node with the given configuration"""
        self.config = config
        self.reasoning_chain = self.config.prompt | self.config.llm.with_structured_output(ReasoningOutput)

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform reasoning on the current state and decide the next action.

        Args:
            state (Dict[str, Any]): The current state of the graph

        Returns:
            Dict[str, Any]: Updated state with reasoning output
        """
        try:
            result = await self.reasoning_chain.ainvoke({
                "state": state.get("context", {}),
                "last_result": state["messages"][-1].content if state.get("messages") else "No previous actions"
            })

            return {
                "reasoning": result,
                "messages": state.get("messages", []) + [{
                    "role": "system",
                    "content": f"Reasoning: {result.analysis}\nNext action: {result.next_action}\nExplanation: {result.explanation}"
                }]
            }
        except Exception as e:
            # Log the error and return a failure state
            print(f"Error in reasoning: {str(e)}")
            return {
                "reasoning": ReasoningOutput(
                    analysis="Error occurred during reasoning",
                    next_action="revise",
                    explanation=f"An error occurred: {str(e)}"
                ),
                "messages": state.get("messages", []) + [{
                    "role": "system",
                    "content": f"Error in reasoning: {str(e)}"
                }]
            }


def create_reasoning_node(config: ReasoningConfig = ReasoningConfig()) -> ToolNode:
    """Create a ToolNode for reasoning"""
    return ToolNode([ReasoningNode(config)])


async def add_reasoning_to_graph(graph: StateGraph, config: ReasoningConfig = ReasoningConfig()):
    """Add the reasoning node to the graph"""
    reasoning_node = create_reasoning_node(config)
    graph.add_node("reason", reasoning_node)

    graph.add_conditional_edges(
        "reason",
        lambda x: x["reasoning"].next_action,
        {
            "continue": "execute_step",
            "revise": "plan_task",
            "finish": "generate_output"
        }
    )


# For testing purposes
async def test_reasoning_node():
    config = ReasoningConfig()
    node = ReasoningNode(config)
    test_state = {
        "context": {"task": "Analyze data"},
        "messages": [{"role": "user", "content": "Please analyze the sales data"}]
    }
    result = await node(test_state)
    assert "reasoning" in result
    assert "messages" in result
    print("Test passed successfully")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_reasoning_node())
