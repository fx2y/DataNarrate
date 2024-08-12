import json
import logging
from typing import Any, Dict, Optional, List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from config import config
from datanarrate.context_manager import ContextManager
from datanarrate.query_analyzer import QueryAnalyzer
from intent_classifier import IntentClassifier
from task_planner import TaskStep, DataSource, TaskPlanner
from tool_selector import ToolSelector


class ToolResult(BaseModel):
    output: Any = Field(description="The output of the tool execution")
    error: Optional[str] = Field(default=None, description="Error message if the execution failed")


class StepResult(BaseModel):
    step_number: int = Field(description="The number of the step in the plan")
    result: ToolResult = Field(description="The result of the tool execution for this step")


class ExecutionEngine:
    def __init__(self, intent_classifier: IntentClassifier, logger: Optional[logging.Logger] = None,
                 max_retries: int = 3):
        self.intent_classifier = intent_classifier
        self.logger = logger or logging.getLogger(__name__)
        self.max_retries = max_retries

    def execute_tool(self, tool: BaseTool, compressed_schema: Dict[str, Any], data_sources: List[DataSource],
                     **kwargs) -> ToolResult:
        """
        Execute a given tool with the provided arguments.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                self.logger.info(f"Executing tool: {tool.name}")
                if tool.name == "SQL Query Tool":
                    optimized_query = self._optimize_sql_query(kwargs['query'], compressed_schema, data_sources)
                    result = tool.invoke({"query": optimized_query})
                elif tool.name == "Elasticsearch Query Tool":
                    optimized_query = self._optimize_es_query(kwargs['query'], compressed_schema, data_sources)
                    result = tool.invoke({"query": optimized_query})
                else:
                    result = tool.invoke(kwargs)
                self.logger.info(f"Tool executed successfully: {tool.name}")
                return ToolResult(output=result)
            except Exception as e:
                retries += 1
                self.logger.error(f"Error executing tool {tool.name}: {e}. Retry {retries}/{self.max_retries}")
                if retries >= self.max_retries:
                    self.logger.error(f"Max retries reached for tool {tool.name}. Failing execution.")
                    return ToolResult(error=str(e))

    def _optimize_sql_query(self, query: str, compressed_schema: Dict[str, Any], data_sources: List[DataSource]) -> str:
        mysql_sources = [ds for ds in data_sources if ds.name == 'mysql']
        if not mysql_sources:
            return query

        # Implement SQL query optimization logic here
        # Use data_sources[0].tables_or_indices and data_sources[0].fields to optimize the query
        # For example:
        # 1. Ensure all relevant tables are included in the FROM clause
        # 2. Limit the SELECT statement to only include suggested fields
        # 3. Add appropriate JOINs based on the relationships between relevant tables
        # 4. Add WHERE clauses to filter data based on the query context
        optimized_query = query  # Placeholder for actual optimization logic
        return optimized_query

    def _optimize_es_query(self, query: Dict[str, Any], compressed_schema: Dict[str, Any],
                           data_sources: List[DataSource]) -> Dict[str, Any]:
        es_sources = [ds for ds in data_sources if ds.name == 'elasticsearch']
        if not es_sources:
            return query

        # Implement Elasticsearch query optimization logic here
        # Use data_sources[0].tables_or_indices (indices in this case) and data_sources[0].fields to optimize the query
        # For example:
        # 1. Ensure the query targets only the relevant indices
        # 2. Limit the fields returned to only the suggested fields
        # 3. Add filters based on the query context
        optimized_query = query  # Placeholder for actual optimization logic
        return optimized_query

    def execute_step(self, step: TaskStep, tool: BaseTool, tool_input: Dict[str, Any],
                     compressed_schema: Dict[str, Any]) -> StepResult:
        """
        Execute a single step from the plan.
        """
        result = self.execute_tool(tool, compressed_schema, step.data_sources, **tool_input)
        return StepResult(step_number=step.step_number, result=result)

    def execute_plan(self, plan: List[TaskStep], tool_selector: ToolSelector, compressed_schema: Dict[str, Any]) -> \
            List[StepResult]:
        """
        Execute a plan consisting of multiple tool executions.
        """
        results = []
        for step in plan:
            tool_and_input = tool_selector.select_tool_for_step(step)
            if not tool_and_input:
                self.logger.error(f"No suitable tool found for step {step.step_number}")
                break
            tool, tool_input = tool_and_input
            result = self.execute_step(step, tool, tool_input, compressed_schema)
            results.append(result)

            if result.result.error:
                self.logger.warning(f"Step {result.step_number} failed: {step}. Stopping plan execution.")
                break

        return results

    def execute(self, query: str, context: dict):
        intent_classification = self.intent_classifier.classify(query)
        if intent_classification.intent == "data_retrieval":
            return self.execute_data_retrieval(query, context)
        elif intent_classification.intent == "visualization":
            return self.execute_visualization(query, context)
        # ... handle other intents ...

    def execute_data_retrieval(self, query: str, context: dict):
        # Implementation for data retrieval
        pass

    def execute_visualization(self, query: str, context: dict):
        # Implementation for visualization
        pass

    # ... other intent-specific execution methods ...


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
    classifier = IntentClassifier(llm)
    context_manager = ContextManager(classifier, thread_id="example_thread")
    query_analyzer = QueryAnalyzer(llm)
    engine = ExecutionEngine(classifier)
    selector = ToolSelector(llm)


    # Example tools
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

        def _run(self, data: Dict[str, Any]) -> str:
            return f"Created visualization for data: {data}"


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
            results = engine.execute_plan(task_plan.steps, selector, compressed_schema)
            for result in results:
                if result.result.error:
                    print(f"Step {result.step_number} failed: {result.result.error}")
                else:
                    print(f"Step {result.step_number} succeeded: {result.result.output}")

        else:
            print("Task planning failed.")
    else:
        print("Query analysis failed.")
