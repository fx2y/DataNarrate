import json
import logging
from typing import Any, Dict, Optional, List

from elasticsearch import Elasticsearch
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from config import config
from datanarrate.context_manager import ContextManager
from datanarrate.query_analyzer import QueryAnalyzer
from intent_classifier import IntentClassifier
from query_generator import QueryGenerator, SQLQuery, ElasticsearchQuery
from query_validator import QueryValidator
from task_planner import TaskStep, DataSource, TaskPlanner, QueryInfo
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
        self.query_generator = QueryGenerator(ChatOpenAI(
            model_name=config.LLM_MODEL_NAME,
            openai_api_base=config.OPENAI_API_BASE,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.2
        ))
        self.query_validator = QueryValidator(logger=logger)
        self.mysql_executor = MySQLExecutor()
        self.elasticsearch_executor = ElasticsearchExecutor()

    def execute_tool(self, tool: BaseTool, compressed_schema: Dict[str, Any], data_sources: List[DataSource],
                     task: str, query_info: Optional[QueryInfo] = None, **kwargs) -> ToolResult:
        """
        Execute a given tool with the provided arguments.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                self.logger.info(f"Executing tool: {tool.name}")

                if tool.name.lower().startswith("sql"):
                    return self.execute_mysql_query(query_info, compressed_schema, **kwargs)
                elif tool.name.lower().startswith("elasticsearch"):
                    return self.execute_elasticsearch_query(query_info, compressed_schema, **kwargs)
                else:
                    # Execute other tools as before
                    result = tool.invoke(kwargs)
                    return ToolResult(output=result)

            except Exception as e:
                self.logger.error(f"Error executing tool {tool.name}: {e}", exc_info=True)
                retries += 1
                if retries >= self.max_retries:
                    return ToolResult(error=f"Max retries reached. Last error: {str(e)}")

    def execute_mysql_query(self, query_info: QueryInfo, compressed_schema: Dict[str, Any], **kwargs) -> ToolResult:
        """
        Execute a MySQL query using the query information provided.
        """
        try:
            sql_query = self.query_generator.generate_sql_query(query_info, compressed_schema['mysql'])
            if not isinstance(sql_query, SQLQuery):
                return ToolResult(error="Failed to generate valid SQLQuery object")

            if not self.query_validator.validate_sql_query(sql_query.query):
                return ToolResult(error="Invalid SQL query generated")

            result = self.mysql_executor.execute_query(sql_query)
            return ToolResult(output=result)
        except Exception as e:
            self.logger.error(f"Error executing MySQL query: {e}", exc_info=True)
            return ToolResult(error=f"MySQL query execution failed: {str(e)}")

    def execute_elasticsearch_query(self, query_info: QueryInfo, compressed_schema: Dict[str, Any],
                                    **kwargs) -> ToolResult:
        """
        Execute an Elasticsearch query using the query information provided.
        """
        try:
            es_query = self.query_generator.generate_elasticsearch_query(query_info, compressed_schema['elasticsearch'])
            if not isinstance(es_query, ElasticsearchQuery):
                return ToolResult(error="Failed to generate valid ElasticsearchQuery object")

            if not self.query_validator.validate_elasticsearch_query(es_query.query):
                return ToolResult(error="Invalid Elasticsearch query generated")

            result = self.elasticsearch_executor.execute_query(es_query)
            return ToolResult(output=result)
        except Exception as e:
            self.logger.error(f"Error executing Elasticsearch query: {e}", exc_info=True)
            return ToolResult(error=f"Elasticsearch query execution failed: {str(e)}")

    def _generate_and_optimize_sql_query(self, task: str, compressed_schema: Dict[str, Any],
                                         data_sources: List[DataSource], query_info: Optional[QueryInfo]) -> SQLQuery:
        mysql_sources = [ds for ds in data_sources if ds.name == 'mysql']
        if not mysql_sources:
            raise ValueError("No MySQL data source found for SQL query generation")

        mysql_schema = compressed_schema.get('mysql', {})
        query_result = self.query_generator.generate_sql_query(task, mysql_schema, query_info)
        if query_result is None:
            raise ValueError("Failed to generate SQL query")

        # Here you can add additional optimization logic if needed
        return query_result

    def _generate_and_optimize_es_query(self, task: str, compressed_schema: Dict[str, Any],
                                        data_sources: List[DataSource],
                                        query_info: Optional[QueryInfo]) -> ElasticsearchQuery:
        es_sources = [ds for ds in data_sources if ds.name == 'elasticsearch']
        if not es_sources:
            raise ValueError("No Elasticsearch data source found for query generation")

        es_schema = compressed_schema.get('elasticsearch', {})
        query_result = self.query_generator.generate_elasticsearch_query(task, es_schema, query_info)
        if query_result is None:
            raise ValueError("Failed to generate Elasticsearch query")

        # Here you can add additional optimization logic if needed
        return query_result

    def execute_step(self, step: TaskStep, tool: BaseTool, tool_input: Dict[str, Any],
                     compressed_schema: Dict[str, Any]) -> StepResult:
        """
        Execute a single step from the plan.
        """
        execute_kwargs = tool_input.copy()  # Create a copy of tool_input to avoid modifying the original

        # Add the task description for all tools
        task = step.description

        # For database-related tools, prepare query_info if it's not None
        query_info = None
        if tool.name in ["SQL Query Tool", "Elasticsearch Query Tool"] and step.query_info is not None:
            query_info = step.query_info

        result = self.execute_tool(tool, compressed_schema, step.data_sources, task=task, query_info=query_info,
                                   **execute_kwargs)
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

            # Ensure the full compressed_schema is passed to execute_tool
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


class MySQLExecutor:
    def __init__(self):
        self.engine = create_engine(config.SQLALCHEMY_DATABASE_URI)
        self.Session = sessionmaker(bind=self.engine)

    def execute_query(self, query: SQLQuery) -> Dict[str, Any]:
        try:
            with self.Session() as session:
                sql_text = text(query.query)
                print(f"Executing SQL query: {sql_text}")  # Debug print
                result = session.execute(sql_text)
                if query.query.strip().lower().startswith('select'):
                    rows = [dict(row) for row in result]
                    print(f"Query returned {len(rows)} rows")  # Debug print
                    return {"result": rows}
                else:
                    session.commit()
                    return {"result": "Query executed successfully"}
        except SQLAlchemyError as e:
            print(f"SQLAlchemy error: {str(e)}")  # Debug print
            raise Exception(f"SQLAlchemy query execution failed: {str(e)}")


class ElasticsearchExecutor:
    def __init__(self):
        self.client = Elasticsearch(
            [config.ELASTICSEARCH_HOST],
            basic_auth=(config.ELASTICSEARCH_USERNAME, config.ELASTICSEARCH_PASSWORD),
            verify_certs=False
        )

    def execute_query(self, query: ElasticsearchQuery) -> Dict[str, Any]:
        try:
            result = self.client.search(index=query.index, body=query.query)
            return {"result": result['hits']['hits']}
        except Exception as e:
            raise Exception(f"Elasticsearch query execution failed: {str(e)}")


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
    query_analyzer = QueryAnalyzer(llm, context_manager)
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
