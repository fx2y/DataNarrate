import logging
from typing import Any, Dict, Optional, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    output: Any = Field(description="The output of the tool execution")
    error: Optional[str] = Field(default=None, description="Error message if the execution failed")


class ExecutionEngine:
    def __init__(self, logger: Optional[logging.Logger] = None, max_retries: int = 3):
        self.logger = logger or logging.getLogger(__name__)
        self.max_retries = max_retries

    def execute_tool(self, tool: BaseTool, **kwargs) -> ToolResult:
        """
        Execute a given tool with the provided arguments.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                self.logger.info(f"Executing tool: {tool.name}")
                result = tool.invoke(kwargs)
                self.logger.info(f"Tool executed successfully: {tool.name}")
                return ToolResult(output=result)
            except Exception as e:
                retries += 1
                self.logger.error(f"Error executing tool {tool.name}: {e}. Retry {retries}/{self.max_retries}")
                if retries >= self.max_retries:
                    self.logger.error(f"Max retries reached for tool {tool.name}. Failing execution.")
                    return ToolResult(error=str(e))

    def execute_plan(self, tools: Dict[str, BaseTool], plan: List[Dict[str, Any]]) -> Dict[str, ToolResult]:
        """
        Execute a plan consisting of multiple tool executions.
        """
        results = {}
        for step in plan:
            tool_name = step.get("tool")
            tool_input = step.get("tool_input", {})

            tool = tools.get(tool_name)
            if not tool:
                self.logger.error(f"Tool {tool_name} not found for step {step}")
                results[tool_name] = ToolResult(error=f"Tool {tool_name} not found")
                continue

            result = self.execute_tool(tool, **tool_input)
            results[tool_name] = result

            if result.error:
                self.logger.warning(f"Step failed: {step}. Stopping plan execution.")
                break

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ExecutionEngine()


    # Example tools
    class SQLQueryTool(BaseTool):
        name = "SQL Query Tool"
        description = "Executes SQL queries on a database"

        def _run(self, query: str) -> str:
            return f"Executed SQL query: {query}"


    class VisualizationTool(BaseTool):
        name = "Visualization Tool"
        description = "Creates data visualizations and charts"

        def _run(self, data: Dict[str, Any]) -> str:
            return f"Created visualization for data: {data}"


    tools = {
        "SQL Query Tool": SQLQueryTool(),
        "Visualization Tool": VisualizationTool()
    }

    plan = [
        {"tool": "SQL Query Tool", "tool_input": {"query": "SELECT * FROM sales"}},
        {"tool": "Visualization Tool", "tool_input": {"data": {"sales": 1000}}}
    ]

    results = engine.execute_plan(tools, plan)
    for tool_name, result in results.items():
        if result.error:
            print(f"{tool_name} failed: {result.error}")
        else:
            print(f"{tool_name} succeeded: {result.output}")
