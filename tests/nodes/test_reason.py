import asyncio
# Unit Tests
import unittest
from typing import Dict, Any

from langgraph.graph import StateGraph


# ... (previous code remains the same) ...


class TestReasoningNode(unittest.TestCase):
    def setUp(self):
        self.config = ReasoningConfig()
        self.node = ReasoningNode(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.node.config, ReasoningConfig)
        self.assertIsNotNone(self.node.reasoning_chain)

    async def test_reasoning_output(self):
        test_state = {
            "context": {"task": "Analyze data"},
            "messages": [{"role": "user", "content": "Please analyze the sales data"}]
        }
        result = await self.node(test_state)
        self.assertIn("reasoning", result)
        self.assertIn("messages", result)
        self.assertIsInstance(result["reasoning"], ReasoningOutput)
        self.assertIn(result["reasoning"].next_action, ["continue", "revise", "finish"])

    async def test_error_handling(self):
        # Simulate an error by providing invalid state
        invalid_state = {}
        result = await self.node(invalid_state)
        self.assertIn("reasoning", result)
        self.assertEqual(result["reasoning"].next_action, "revise")
        self.assertIn("Error", result["reasoning"].explanation)


# Integration Test
async def test_graph_integration():
    graph = StateGraph(Dict[str, Any])
    config = ReasoningConfig()
    await add_reasoning_to_graph(graph, config)

    # Compile the graph
    workflow = graph.compile()

    # Test the graph with a sample input
    input_state = {
        "context": {"task": "Analyze market trends"},
        "messages": [{"role": "user", "content": "What are the current market trends?"}]
    }

    result = await workflow.ainvoke(input_state)

    assert "reasoning" in result
    assert result["reasoning"].next_action in ["continue", "revise", "finish"]
    print("Graph integration test passed successfully")


# Demonstration Script
async def demonstrate_reasoning_node():
    print("Demonstrating ReasoningNode functionality:")

    # Create a sample state
    sample_state = {
        "context": {"task": "Predict future sales"},
        "messages": [
            {"role": "user", "content": "Can you help me predict our future sales?"},
            {"role": "assistant",
             "content": "Certainly! I'd be happy to help you predict future sales. To get started, I'll need some information about your current sales data and any relevant factors that might influence future sales. Could you please provide me with the following details:\n\n1. Your sales data for the past 12 months (monthly breakdown if possible)\n2. Any seasonal trends you've noticed in your sales\n3. Information about your product line (any new products or discontinuations)\n4. Any major marketing campaigns planned\n5. Economic factors that might impact your industry\n\nOnce I have this information, I can analyze the data and help you make predictions about future sales trends."},
            {"role": "user",
             "content": "Here's a summary of our data:\n1. Past 12 months sales: Jan: $100k, Feb: $120k, Mar: $150k, Apr: $130k, May: $140k, Jun: $160k, Jul: $180k, Aug: $200k, Sep: $190k, Oct: $210k, Nov: $230k, Dec: $250k\n2. We see higher sales in summer and holiday seasons\n3. We introduced a new product line in June\n4. We're planning a major marketing campaign for next quarter\n5. The economy is generally stable, with a slight growth trend in our industry"}
        ]
    }

    # Create and run the reasoning node
    config = ReasoningConfig()
    node = ReasoningNode(config)
    result = await node(sample_state)

    print("\nReasoning Output:")
    print(f"Analysis: {result['reasoning'].analysis}")
    print(f"Next Action: {result['reasoning'].next_action}")
    print(f"Explanation: {result['reasoning'].explanation}")

    print("\nUpdated Messages:")
    for message in result['messages']:
        print(f"{message['role']}: {message['content']}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run integration test
    asyncio.run(test_graph_integration())

    # Run demonstration
    asyncio.run(demonstrate_reasoning_node())
