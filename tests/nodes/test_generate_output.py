import asyncio
import json

from langchain_core.runnables import RunnableConfig
from nodes.generate_output import generate_output, OutputState, OutputFormat


async def test_generate_output():
    # Mock analysis results
    mock_analysis_results = """
    The data analysis of FooBar company reveals the following:
    1. The company raised 1 Billion dollars in funding recently.
    2. They are now focusing on hiring AI specialists.
    3. FooBar was founded in 2019, making it a relatively young company.
    4. Their main product line consists of friendly robots.
    """

    # Create a mock state
    mock_state = OutputState(analysis_results=mock_analysis_results)

    # Create a mock config
    mock_config = RunnableConfig(callbacks=None)

    # Run the generate_output function
    result = await generate_output(mock_state, mock_config)

    # Assert that the output is present and has the correct structure
    assert "output" in result
    assert isinstance(result["output"], OutputFormat)
    assert "messages" in result
    assert len(result["messages"]) == 1

    # Parse the message content (which should be a JSON string)
    message_content = json.loads(result["messages"][0].content)

    # Assert that all expected fields are present in the output
    assert "summary" in message_content
    assert "key_points" in message_content
    assert "insights" in message_content
    assert "visualizations" in message_content
    assert "next_steps" in message_content

    # Print the output for manual inspection
    print("Generated Output:")
    print(json.dumps(message_content, indent=2))

    # Additional specific assertions can be added here based on expected content


if __name__ == "__main__":
    asyncio.run(test_generate_output())
