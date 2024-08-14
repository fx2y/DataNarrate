import json
import logging
from typing import Any, Dict, Optional, List, Union

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from scipy import stats
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from config import config
from datanarrate.context_manager import ContextManager
from datanarrate.query_analyzer import QueryAnalyzer
from intent_classifier import IntentClassifier
from query_generator import QueryGenerator, SQLQuery, ElasticsearchQuery
from query_validator import QueryValidator
from task_planner import TaskStep, DataSource, QueryInfo
from tool_selector import ToolSelector
from visualization_generator import VisualizationGenerator


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
        self.visualization_generator = VisualizationGenerator(self.intent_classifier.llm, logger=self.logger)
        self.data_analysis_tool = DataAnalysisTool()

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
                elif tool.name == "Visualization Tool":
                    return self._execute_visualization_tool(tool, compressed_schema, data_sources, task, query_info,
                                                            **kwargs)
                elif tool.name == "Data Analysis Tool":
                    return self._execute_data_analysis_tool(task, **kwargs)
                else:
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

            self.logger.info(f"Executing MySQL query: {sql_query.query}")  # Add this line
            result = self.mysql_executor.execute_query(sql_query)
            self.logger.info(f"MySQL query result: {result}")  # Add this line
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

    def _execute_visualization_tool(self, tool: BaseTool, compressed_schema: Dict[str, Any],
                                    data_sources: List[DataSource],
                                    task: str, query_info: Optional[QueryInfo] = None, **kwargs) -> ToolResult:
        try:
            self.logger.info(f"Generating visualization for task: {task}")
            viz_spec = self.visualization_generator.generate_visualization(
                data=kwargs.get("data", {}),
                requirements=task,
                user_preferences=kwargs.get("user_preferences", {})
            )
            if viz_spec and self.visualization_generator.validate_spec(viz_spec):
                self.logger.info(f"Generated valid visualization specification for task: {task}")
                return ToolResult(output=viz_spec)
            else:
                return ToolResult(error="Failed to generate a valid visualization specification")
        except Exception as e:
            self.logger.error(f"Error generating visualization for task {task}: {e}", exc_info=True)
            return ToolResult(error=f"Error generating visualization: {str(e)}")

    def _execute_data_analysis_tool(self, task: str, **kwargs) -> ToolResult:
        try:
            self.logger.info(f"Performing data analysis for task: {task}")
            analysis_result = self.data_analysis_tool.analyze(task, **kwargs)
            return ToolResult(output=analysis_result)
        except Exception as e:
            self.logger.error(f"Error performing data analysis for task {task}: {e}", exc_info=True)
            return ToolResult(error=f"Error performing data analysis: {str(e)}")

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
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result]
                    print(f"Query returned {len(rows)} rows")  # Debug print
                    return {"result": rows}
                else:
                    session.commit()
                    return {"result": "Query executed successfully"}
        except SQLAlchemyError as e:
            print(f"SQLAlchemy error: {str(e)}")  # Debug print
            raise Exception(f"SQLAlchemy query execution failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")  # Debug print
            raise Exception(f"Unexpected error during query execution: {str(e)}")


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


class DataAnalysisTool:
    def analyze(self, task: str, data: Union[pd.DataFrame, Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        if analysis_type == "descriptive":
            return self._descriptive_analysis(data)
        elif analysis_type == "correlation":
            return self._correlation_analysis(data)
        elif analysis_type == "hypothesis_test":
            return self._hypothesis_test(data, task)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

    def _descriptive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {
            "summary": data.describe().to_dict(),
            "column_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict()
        }

    def _correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr().to_dict()
        return {"correlation_matrix": correlation_matrix}

    def _hypothesis_test(self, data: pd.DataFrame, task: str) -> Dict[str, Any]:
        # This is a simplified example. In practice, you'd need to parse the task
        # to determine which columns to use and what type of test to perform.
        if "compare" in task.lower() and len(data.columns) >= 2:
            col1, col2 = data.columns[:2]
            t_stat, p_value = stats.ttest_ind(data[col1], data[col2])
            return {
                "test_type": "Independent t-test",
                "t_statistic": t_stat,
                "p_value": p_value
            }
        else:
            raise ValueError("Unsupported hypothesis test task")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=config.LOG_LEVEL)

    # Initialize components
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


    # Example tools (these should match the tools in PlanAndExecute)
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

        def _run(self, data: Dict[str, Any], requirements: str, user_preferences: Dict[str, Any]) -> str:
            return f"Created visualization for data: {data}, requirements: {requirements}, preferences: {user_preferences}"


    class DataAnalysisTool(BaseTool):
        name = "Data Analysis Tool"
        description = "Performs statistical analysis on datasets"

        def _run(self, dataset: str, analysis_type: str) -> str:
            return f"Performed {analysis_type} analysis on {dataset}"


    # Register tools
    selector.register_tool(SQLQueryTool())
    selector.register_tool(ElasticsearchQueryTool())
    selector.register_tool(VisualizationTool())
    selector.register_tool(DataAnalysisTool())

    # Example compressed schema (this should match the schema in PlanAndExecute)
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

    # Example usage of ExecutionEngine
    print("ExecutionEngine Example:")

    # Example task step
    task_step = TaskStep(
        step_number=1,
        description="Retrieve top 5 selling products in Q2",
        required_capability="SQL Query",
        tools=["SQL Query Tool"],
        data_sources=[DataSource(
            name="mysql",
            tables_or_indices=["sales", "products"],
            fields={
                "sales": ["date", "product_id", "revenue"],
                "products": ["name"]
            }
        )],
        query_info=QueryInfo(
            query_type="SELECT",
            data_source="mysql",
            tables_or_indices=["sales", "products"],
            fields=["sales.date", "sales.product_id", "sales.revenue", "products.name"],
            columns=["products.name", "SUM(sales.revenue) as total_revenue"],
            conditions="sales.date BETWEEN '2023-04-01' AND '2023-06-30'",
            group_by=["products.name"],
            order_by=["total_revenue DESC"],
            limit=5
        )
    )

    # Select tool for the step
    tool_and_input = selector.select_tool_for_step(task_step)
    if tool_and_input:
        tool, tool_input = tool_and_input

        # Execute the step
        result = engine.execute_step(task_step, tool, tool_input, compressed_schema)

        print(f"Step {result.step_number} result:")
        if result.result.error:
            print(f"Error: {result.result.error}")
        else:
            print(f"Output: {result.result.output}")
    else:
        print("No suitable tool found for the step.")

    print(
        "\nNote: This is a simplified example. In practice, execution is managed by PlanAndExecute in plan_execute.py")
