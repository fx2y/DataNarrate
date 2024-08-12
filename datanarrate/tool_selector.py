import json
import logging
import os
from typing import Dict, Optional, Any, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from task_planner import TaskStep


class ToolSelection(BaseModel):
    tool_name: str = Field(description="The name of the selected tool")
    reason: str = Field(description="Reason for selecting this tool")
    tool_input: Dict[str, Any] = Field(description="Input parameters for the tool")


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
                       "Consider the task description, required capability, and input description. "
                       "Provide the tool name, a brief reason for your selection, and the appropriate tool input based on the tool's schema. "
                       "Selection format: {format_instructions}"),
            ("human",
             "Task: {task}\nRequired capability: {required_capability}\nInput description: {input_description}\nAvailable tools: {tools}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def register_tool(self, tool: BaseTool):
        """Register a new tool in the selector."""
        self.tool_registry[tool.name] = tool
        self.logger.info(f"Registered new tool: {tool.name}")

    def get_tool_descriptions(self) -> str:
        """Get a string of all registered tool names, descriptions, and schemas."""
        descriptions = []
        for name, tool in self.tool_registry.items():
            schema = self.get_tool_schema(name)
            description = f"{name}: {tool.description}\nSchema: {json.dumps(schema, indent=2)}"
            descriptions.append(description)
        return "\n\n".join(descriptions)

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        tool = self.tool_registry.get(tool_name)
        if tool and hasattr(tool, 'args'):
            return tool.args
        return {}

    def select_tool_for_step(self, step: TaskStep) -> Optional[Tuple[BaseTool, Dict[str, Any]]]:
        try:
            self.logger.info(f"Selecting tool for step: {step.description}")
            tool_descriptions = self.get_tool_descriptions()
            selection = self.selection_chain.invoke({
                "task": step.description,
                "required_capability": step.required_capability,
                "input_description": json.dumps(step.input_description),
                "tools": tool_descriptions
            })
            self.logger.info(f"Selected tool: {selection.tool_name}")
            self.logger.debug(f"Selection reason: {selection.reason}")
            self.logger.debug(f"Tool input: {selection.tool_input}")
            tool = self.tool_registry.get(selection.tool_name)
            return (tool, selection.tool_input) if tool else None
        except Exception as e:
            self.logger.error(f"Error selecting tool: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    selector = ToolSelector("deepseek-chat", openai_api_base='https://api.deepseek.com',
                            openai_api_key=os.environ["DEEPSEEK_API_KEY"])


    # Register some example tools
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


    class DataAnalysisTool(BaseTool):
        name = "Data Analysis Tool"
        description = "Performs statistical analysis on datasets"

        def _run(self, dataset: str, analysis_type: str) -> str:
            return f"Performed {analysis_type} analysis on {dataset}"


    selector.register_tool(SQLQueryTool())
    selector.register_tool(VisualizationTool())
    selector.register_tool(DataAnalysisTool())

    # Test the tool selector
    example_steps = [
        TaskStep(
            step_number=1,
            description="Retrieve the sales data for Q2 from our database",
            required_capability="data_query",
            input_description={"query": "Sales data for Q2"}
        ),
        TaskStep(
            step_number=2,
            description="Analyze the top 10 products by revenue",
            required_capability="data_analysis",
            input_description={"dataset": "Q2 sales data", "analysis_type": "top performers"}
        ),
        TaskStep(
            step_number=3,
            description="Create a bar chart of the top 10 products by revenue",
            required_capability="data_visualization",
            input_description={"data": "Top 10 products by revenue"}
        )
    ]

    for step in example_steps:
        result = selector.select_tool_for_step(step)
        if result:
            tool, tool_input = result
            print(f"Step {step.step_number}: {step.description}")
            print(f"Selected Tool: {tool.name}")
            print(f"Tool Input: {tool_input}")
            print("---")
        else:
            print(f"Failed to select tool for step: {step.description}")
            print("---")
