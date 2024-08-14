import json
import logging
from typing import Dict, Optional, Any, Tuple, List

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
from task_planner import TaskStep, TaskPlanner, DataSource, QueryInfo


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
            ("system",
             "You are a tool selection expert. Your task is to select the most appropriate tool for a given step in a data analysis task. "
             "Consider the step description, required capability, and available tools. "
             "Provide the tool name, a brief reason for your selection, and the appropriate tool input based on the tool's schema. "
             "Output format: {format_instructions}"),
            ("human", "Step description: {step_description}\n"
                      "Required capability: {required_capability}\n"
                      "Data sources: {data_sources}\n"
                      "Query information: {query_info}\n"
                      "Previous results: {previous_results}\n"
                      "Available tools: {tools}\n\n"
                      "Select the most appropriate tool and provide the necessary input.")
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

    def select_tool_for_step(self, step: TaskStep, previous_results: List[Dict[str, Any]] = None) -> Optional[
        Tuple[BaseTool, Dict[str, Any]]]:
        tools_description = self.get_tool_descriptions()
        data_sources_info = self._format_data_sources_info(step.data_sources)
        query_info = self._format_query_info(step.query_info)
        previous_results_info = self._format_previous_results(previous_results)

        try:
            selection = self.selection_chain.invoke({
                "step_description": step.description,
                "required_capability": step.required_capability,
                "data_sources": data_sources_info,
                "query_info": query_info,
                "previous_results": previous_results_info,
                "tools": tools_description
            })

            selected_tool = self.tool_registry.get(selection.tool_name)
            if selected_tool:
                self.logger.info(f"Selected tool: {selection.tool_name}. Reason: {selection.reason}")
                return selected_tool, selection.tool_input
            else:
                self.logger.warning(f"Selected tool '{selection.tool_name}' not found in registry.")
                return None
        except Exception as e:
            self.logger.error(f"Error in tool selection: {e}")
            return None

    def _format_data_sources_info(self, data_sources: List[DataSource]) -> str:
        if not data_sources:
            return "No specific data sources provided."

        info = []
        for ds in data_sources:
            fields_info = ", ".join([f"{table}: {', '.join(fields)}" for table, fields in ds.fields.items()])
            info.append(f"- {ds.name} (Tables/Indices: {', '.join(ds.tables_or_indices)}; Fields: {fields_info})")

        return "\n".join(info)

    def _format_query_info(self, query_info: Optional[QueryInfo]) -> str:
        if not query_info:
            return "No specific query information provided."

        fields_info = (
            ', '.join(query_info.fields) if isinstance(query_info.fields, list)
            else ', '.join([f"{table}: {', '.join(fields)}" for table, fields in query_info.fields.items()])
        )

        return f"""
- Data Source: {query_info.data_source}
- Query Type: {query_info.query_type}
- Tables/Indices: {', '.join(query_info.tables_or_indices)}
- Fields: {fields_info}
- Conditions: {query_info.filters if query_info.filters else 'Not specified'}
- Group By: {', '.join(query_info.aggregations) if query_info.aggregations else 'Not specified'}
- Order By: {', '.join(query_info.order_by) if query_info.order_by else 'Not specified'}
- Limit: {query_info.limit if query_info.limit else 'Not specified'}
"""

    def _format_previous_results(self, previous_results: List[Dict[str, Any]]) -> str:
        if not previous_results:
            return "No previous results available."

        formatted_results = []
        for i, result in enumerate(previous_results, 1):
            formatted_results.append(f"Step {i} Result: {result}")

        return "\n".join(formatted_results)


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
    query_analyzer = QueryAnalyzer(llm, context_manager)
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
