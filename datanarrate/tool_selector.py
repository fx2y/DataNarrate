import logging
import os
from typing import Dict, Optional, Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


class ToolSelection(BaseModel):
    tool_name: str = Field(description="The name of the selected tool")
    reason: str = Field(description="Reason for selecting this tool")


class ToolSelector:
    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None, **kwargs):
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        self.llm = self._create_llm(model_name, **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=ToolSelection)
        self.tool_registry: Dict[str, BaseTool] = {}
        self.selection_chain = self._create_selection_chain()

    def _create_llm(self, model_name: str, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(model_name=model_name, temperature=0.2, **kwargs)

    def _create_selection_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Select the most appropriate tool for the given task. "
                       "Consider the task description and available tools. "
                       "Provide the tool name and a brief reason for your selection. "
                       "Selection format: {format_instructions}"),
            ("human", "Task: {task}\nAvailable tools: {tools}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def register_tool(self, tool: BaseTool):
        """Register a new tool in the selector."""
        self.tool_registry[tool.name] = tool
        self.logger.info(f"Registered new tool: {tool.name}")

    def get_tool_descriptions(self) -> str:
        """Get a string of all registered tool names and descriptions."""
        return "\n".join([f"{name}: {tool.description}" for name, tool in self.tool_registry.items()])

    def select_tool(self, task: str) -> Optional[BaseTool]:
        try:
            self.logger.info(f"Selecting tool for task: {task}")
            tool_descriptions = self.get_tool_descriptions()
            selection = self.selection_chain.invoke({
                "task": task,
                "tools": tool_descriptions
            })
            self.logger.info(f"Selected tool: {selection.tool_name}")
            self.logger.debug(f"Selection reason: {selection.reason}")
            return self.tool_registry.get(selection.tool_name)
        except Exception as e:
            self.logger.error(f"Error selecting tool: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    selector = ToolSelector("deepseek-coder", openai_api_base='https://api.deepseek.com',
                            openai_api_key=os.environ["DEEPSEEK_API_KEY"])

    main_logger = logging.getLogger(__name__)


    # Register some example tools
    class SQLQueryTool(BaseTool):
        def _run(self, *args: Any, **kwargs: Any) -> Any:
            main_logger.info(f"{self.name}: {self.description}")

        name = "SQL Query Tool"
        description = "Executes SQL queries on a database"


    class VisualizationTool(BaseTool):
        def _run(self, *args: Any, **kwargs: Any) -> Any:
            main_logger.info(f"{self.name}: {self.description}")

        name = "Visualization Tool"
        description = "Creates data visualizations and charts"


    class StorytellingTool(BaseTool):
        def _run(self, *args: Any, **kwargs: Any) -> Any:
            main_logger.info(f"{self.name}: {self.description}")

        name = "Storytelling Tool"
        description = "Generates narrative insights from data analysis"


    selector.register_tool(SQLQueryTool())
    selector.register_tool(VisualizationTool())
    selector.register_tool(StorytellingTool())

    # Test the tool selector
    tasks = [
        "Retrieve the sales data for Q2 from our database",
        "Create a bar chart of our top 10 products by revenue",
        "Generate a report explaining the trends in our customer acquisition over the last year"
    ]

    for task in tasks:
        selected_tool = selector.select_tool(task)
        if selected_tool:
            print(f"Task: {task}")
            print(f"Selected Tool: {selected_tool.name}")
            print("---")
        else:
            print(f"Failed to select tool for task: {task}")
            print("---")
