import pytest
from langchain_core.tools import BaseTool as LangChainBaseTool
from tools import BaseTool, ToolRegistry, register_tool, get_tool, list_tools


class SimpleTool(BaseTool):
    def _run(self, input: str) -> str:
        return f"Processed: {input}"

    async def _arun(self, input: str) -> str:
        return f"Async Processed: {input}"


def test_base_tool():
    tool = SimpleTool(name="simple", description="A simple tool")
    result = tool.run(input="test")
    assert result.output == "Processed: test"
    assert result.error is None


@pytest.mark.asyncio
async def test_base_tool_async():
    tool = SimpleTool(name="simple", description="A simple tool")
    result = await tool.arun(input="test")
    assert result.output == "Async Processed: test"
    assert result.error is None


def test_tool_registry():
    registry = ToolRegistry()
    tool = SimpleTool(name="simple", description="A simple tool")
    registry.register(tool)
    assert registry.get("simple") == tool
    assert "simple" in registry.list_tools()


def test_global_registry():
    tool = SimpleTool(name="global_simple", description="A global simple tool")
    register_tool(tool)
    assert get_tool("global_simple") == tool
    assert "global_simple" in list_tools()


def test_langchain_tool_integration():
    def lc_tool_func(x: str) -> str:
        return f"LangChain: {x}"

    lc_tool = LangChainBaseTool(
        name="lc_tool",
        description="A LangChain tool",
        func=lc_tool_func
    )
    register_tool(lc_tool)
    wrapped_tool = get_tool("lc_tool")
    assert wrapped_tool is not None
    result = wrapped_tool.run(x="test")
    assert result.output == "LangChain: test"


def test_to_langchain_tool():
    tool = SimpleTool(name="convertible", description="A convertible tool")
    lc_tool = tool.to_langchain_tool()
    assert isinstance(lc_tool, LangChainBaseTool)
    assert lc_tool.name == "convertible"
    assert lc_tool.description == "A convertible tool"
    assert lc_tool.run(input="test") == "Processed: test"
