from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool as LangChainBaseTool
from langchain_core.pydantic_v1 import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized output format for tool execution."""
    output: Optional[Any] = Field(None, description="The output of the tool execution")
    error: Optional[str] = Field(None, description="Error message if the tool execution failed")


class BaseTool(ABC, BaseModel):
    """Base class for all tools in the DataNarration System."""
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="A description of what the tool does")
    return_direct: bool = Field(False, description="Whether to return the tool output directly")
    verbose: bool = Field(False, description="Whether to print out the progress of the tool")
    callbacks: Optional[CallbackManagerForToolRun] = Field(None, description="Callbacks for the tool")

    @root_validator
    def check_return_direct(cls, values):
        """Ensure return_direct is False if callbacks are provided."""
        if values.get("return_direct") and values.get("callbacks"):
            raise ValueError("return_direct cannot be True if callbacks are provided.")
        return values

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool's core functionality."""
        pass

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool's core functionality asynchronously."""
        return await self._run(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Run the tool with error handling and return a standardized result."""
        try:
            result = self._run(*args, **kwargs)
            return ToolResult(output=result)
        except Exception as e:
            return ToolResult(error=f"{type(e).__name__}: {str(e)}")

    async def arun(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Run the tool asynchronously with error handling and return a standardized result."""
        try:
            result = await self._arun(*args, **kwargs)
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
            return_direct=self.return_direct,
            verbose=self.verbose,
        )


class ToolRegistry:
    """Registry for managing tools in the DataNarration System."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: Union[BaseTool, LangChainBaseTool]) -> None:
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

    def get_tool_names(self) -> List[str]:
        """Get a list of all registered tool names."""
        return list(self._tools.keys())


class LangChainToolWrapper(BaseTool):
    """Wrapper for LangChain tools to make them compatible with our BaseTool."""
    _tool: LangChainBaseTool

    def __init__(self, tool: LangChainBaseTool):
        super().__init__(
            name=tool.name,
            description=tool.description,
            return_direct=tool.return_direct,
            verbose=tool.verbose
        )
        self._tool = tool

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self._tool.run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self._tool.arun(*args, **kwargs)


tool_registry = ToolRegistry()
