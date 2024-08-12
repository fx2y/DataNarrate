import json
import logging
from typing import Dict, Optional, Any, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from config import config
from context_manager import ContextManager
from datanarrate.intent_classifier import IntentClassifier
from datanarrate.query_analyzer import QueryAnalyzer
from task_planner import TaskStep, TaskPlanner


class ToolSelection(BaseModel):
    tool_name: str = Field(description="The name of the selected tool")
    reason: str = Field(description="Reason for selecting this tool")
    tool_input: Dict[str, Any] = Field(description="Input parameters for the tool")


class ToolSelector:
    def __init__(self, llm: BaseChatModel, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=ToolSelection)
        self.tool_registry: Dict[str, BaseTool] = {}
        self.selection_chain = self._create_selection_chain()

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
    # Set up logging
    logging.basicConfig(level=config.LOG_LEVEL)

    # Initialize LLM
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.2
    )

    # Initialize components
    intent_classifier = IntentClassifier(llm)
    context_manager = ContextManager(intent_classifier, thread_id="example_thread")
    query_analyzer = QueryAnalyzer(llm)
    selector = ToolSelector(llm)


    # Register tools
    class SQLQueryTool(BaseTool):
        name = "SQL Query Tool"
        description = "Executes SQL queries on a MySQL database"

        def _run(self, query: str) -> str:
            return f"Executed SQL query: {query}"


    class ElasticsearchQueryTool(BaseTool):
        name = "Elasticsearch Query Tool"
        description = "Executes queries on Elasticsearch indices"

        def _run(self, query: Dict[str, Any]) -> str:
            return f"Executed Elasticsearch query: {json.dumps(query)}"


    class VisualizationTool(BaseTool):
        name = "Visualization Tool"
        description = "Creates data visualizations and charts"

        def _run(self, data: Dict[str, Any], chart_type: str) -> str:
            return f"Created {chart_type} visualization for data: {json.dumps(data)}"


    class DataAnalysisTool(BaseTool):
        name = "Data Analysis Tool"
        description = "Performs statistical analysis on datasets"

        def _run(self, dataset: str, analysis_type: str) -> str:
            return f"Performed {analysis_type} analysis on {dataset}"


    selector.register_tool(SQLQueryTool())
    selector.register_tool(ElasticsearchQueryTool())
    selector.register_tool(VisualizationTool())
    selector.register_tool(DataAnalysisTool())

    # Test the tool selector with a realistic scenario
    test_query = "Show me a bar chart of our top 5 selling products in Q2, including their revenue and compare it with last year's Q2 performance"

    # Update context with the test query
    context_manager.update_context(test_query)

    compressed_schema = {
        "mysql": {
            "sales": ["date:dat", "product_id:int*", "customer_id:int", "quantity:int", "revenue:dec"],
            "products": ["product_id:int*", "name:var", "category:var", "price:dec"],
            "customers": ["customer_id:int*", "name:var", "location:var"]
        },
        "elasticsearch": {
            "orders_index": {
                "order_id": "key",
                "date": "dat",
                "customer": "nes",
                "customer.id": "key",
                "customer.name": "tex",
                "customer.email": "tex",
                "items": "nes",
                "items.product_id": "key",
                "items.name": "tex",
                "items.quantity": "int",
                "items.price": "flo",
                "total_amount": "flo"
            },
            "product_index": {
                "product_id": "key",
                "name": "tex",
                "category": "key",
                "price": "flo",
                "specifications": "nes",
                "specifications.brand": "tex",
                "specifications.model": "tex",
                "specifications.year": "int"
            }
        }
    }

    # Analyze the query
    query_analysis = query_analyzer.analyze_query(
        test_query,
        context_manager.get_state().current_intents,
        compressed_schema
    )

    if query_analysis:
        # Create a task plan
        task_planner = TaskPlanner(llm, context_manager)
        task_plan = task_planner.plan_task(query_analysis, compressed_schema)

        if task_plan:
            print("Task Plan:")
            for step in task_plan.steps:
                print(f"\nStep {step.step_number}: {step.description}")
                print(f"Required Capability: {step.required_capability}")

                # Select tool for the step
                tool_selection = selector.select_tool_for_step(step)
                if tool_selection:
                    selected_tool, tool_input = tool_selection
                    print(f"Selected Tool: {selected_tool.name}")
                    print(f"Tool Input: {json.dumps(tool_input, indent=2)}")
                else:
                    print("Failed to select a tool for this step")

                print("Data Sources:")
                for data_source in step.data_sources:
                    print(f"  - {data_source.name}:")
                    print(f"    Tables/Indices: {', '.join(data_source.tables_or_indices)}")
                    print(f"    Fields: {json.dumps(data_source.fields, indent=2)}")
        else:
            print("Task planning failed.")
    else:
        print("Query analysis failed.")
