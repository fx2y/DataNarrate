from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized output format for tool execution."""
    output: Optional[Any] = None
    error: Optional[str] = None


class BaseTool(ABC, BaseModel):
    """Base class for all tools in the DataNarration System."""
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="A description of what the tool does")

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Execute the tool's core functionality."""
        pass

    async def _arun(self, **kwargs: Any) -> Any:
        """Execute the tool's core functionality asynchronously."""
        return await self._run(**kwargs)

    def run(self, **kwargs: Any) -> ToolResult:
        """Run the tool with error handling and return a standardized result."""
        try:
            result = self._run(**kwargs)
            return ToolResult(output=result)
        except Exception as e:
            return ToolResult(error=f"{type(e).__name__}: {str(e)}")

    async def arun(self, **kwargs: Any) -> ToolResult:
        """Run the tool asynchronously with error handling and return a standardized result."""
        try:
            result = await self._arun(**kwargs)
            return ToolResult(output=result)
        except Exception as e:
            return ToolResult(error=f"{type(e).__name__}: {str(e)}")

    def to_langchain_tool(self) -> LangChainBaseTool:
        """Convert this tool to a LangChain-compatible tool."""
        return LangChainBaseTool(
            name=self.name,
            description=self.description,
            func=self._run,
            coroutine=self._arun,
        )


class ToolRegistry:
    """Registry for managing tools in the DataNarration System."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: Union[BaseTool, LangChainBaseTool]):
        """Register a new tool, supporting both custom and LangChain tools."""
        if isinstance(tool, LangChainBaseTool):
            tool = LangChainToolWrapper(tool)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all registered tools."""
        return {name: {"name": tool.name, "description": tool.description} for name, tool in self._tools.items()}


class LangChainToolWrapper(BaseTool):
    """Wrapper for LangChain tools to make them compatible with our BaseTool."""
    _tool: LangChainBaseTool

    def __init__(self, tool: LangChainBaseTool):
        super().__init__(name=tool.name, description=tool.description)
        self._tool = tool

    def _run(self, **kwargs: Any) -> Any:
        return self._tool.run(**kwargs)

    async def _arun(self, **kwargs: Any) -> Any:
        return await self._tool.arun(**kwargs)


tool_registry = ToolRegistry()
