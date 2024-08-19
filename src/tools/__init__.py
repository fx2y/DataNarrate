from langchain_core.tools import BaseTool as LangChainBaseTool

from .base import BaseTool, ToolRegistry, tool_registry, LangChainToolWrapper


def register_tool(tool: Union[BaseTool, LangChainBaseTool]):
    """Register a tool with the global registry."""
    tool_registry.register(tool)


def get_tool(name: str) -> Optional[BaseTool]:
    """Get a tool from the global registry."""
    return tool_registry.get(name)


def list_tools() -> Dict[str, Dict[str, Any]]:
    """List all registered tools."""
    return tool_registry.list_tools()
