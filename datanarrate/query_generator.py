import logging
from typing import Dict, Any, Optional, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config


class SQLQuery(BaseModel):
    query: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Explanation of the generated query")


class ElasticsearchQuery(BaseModel):
    query: Dict[str, Any] = Field(description="The generated Elasticsearch query")
    explanation: str = Field(description="Explanation of the generated query")


class QueryGenerator:
    def __init__(self, llm: BaseChatModel, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.sql_output_parser = PydanticOutputParser(pydantic_object=SQLQuery)
        self.es_output_parser = PydanticOutputParser(pydantic_object=ElasticsearchQuery)
        self.sql_chain = self._create_sql_chain()
        self.es_chain = self._create_es_chain()

    def _create_sql_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert SQL query generator. Given a task description and schema information, "
                       "generate an appropriate SQL query. Ensure the query is efficient and follows best practices. "
                       "Output format: {format_instructions}"),
            ("human", "Task: {task}\nSchema: {schema}\nGenerate an SQL query for this task.")
        ]).partial(format_instructions=self.sql_output_parser.get_format_instructions())
        return prompt | self.llm | self.sql_output_parser

    def _create_es_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Elasticsearch query generator. Given a task description and index mapping, "
                       "generate an appropriate Elasticsearch query. Ensure the query is efficient and follows best practices. "
                       "Provide both the query and an explanation of how it works. "
                       "Output format: {format_instructions}"),
            ("human", "Task: {task}\nIndex Mapping: {mapping}\nGenerate an Elasticsearch query for this task.")
        ]).partial(format_instructions=self.es_output_parser.get_format_instructions())
        return prompt | self.llm | self.es_output_parser

    def generate_sql_query(self, task: str, schema: Dict[str, Any]) -> Optional[SQLQuery]:
        try:
            self.logger.info(f"Generating SQL query for task: {task}")
            query = self.sql_chain.invoke({"task": task, "schema": schema})
            self.logger.info(f"Generated SQL query: {query.query}")
            return query
        except Exception as e:
            self.logger.error(f"Error generating SQL query: {e}", exc_info=True)
            return None

    def generate_elasticsearch_query(self, task: str, mapping: Dict[str, Any]) -> Optional[ElasticsearchQuery]:
        try:
            self.logger.info(f"Generating Elasticsearch query for task: {task}")
            result = self.es_chain.invoke({"task": task, "mapping": mapping})

            # Check if the explanation is missing and generate one if needed
            if not hasattr(result, 'explanation') or not result.explanation:
                self.logger.warning("Explanation missing from LLM output. Generating a default explanation.")
                explanation = self._generate_default_explanation(result.query, task)
                return ElasticsearchQuery(query=result.query, explanation=explanation)

            self.logger.info(f"Generated Elasticsearch query: {result.query}")
            return result
        except OutputParserException as e:
            self.logger.warning(f"OutputParserException: {e}. Attempting to salvage the query.")
            # Try to extract the query from the error message
            query_dict = self._extract_query_from_error(str(e))
            if query_dict:
                explanation = self._generate_default_explanation(query_dict, task)
                return ElasticsearchQuery(query=query_dict, explanation=explanation)
            else:
                self.logger.error("Failed to salvage the query.")
                return None
        except Exception as e:
            self.logger.error(f"Error generating Elasticsearch query: {e}", exc_info=True)
            return None

    def _generate_default_explanation(self, query: Dict[str, Any], task: str) -> str:
        # Generate a simple explanation based on the query structure
        explanation = f"This query addresses the task: '{task}'. "
        if 'bool' in query:
            explanation += "It uses a bool query to apply filters. "
        if 'range' in str(query):
            explanation += "It includes a date range filter. "
        if 'aggs' in query:
            explanation += "It uses aggregations to calculate results. "
        explanation += "Please review the query structure for more details."
        return explanation

    def _extract_query_from_error(self, error_message: str) -> Optional[Dict[str, Any]]:
        import json
        # Try to extract the query part from the error message
        start = error_message.find('{')
        end = error_message.rfind('}')
        if start != -1 and end != -1:
            try:
                query_str = error_message[start:end + 1]
                return json.loads(query_str)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse query from error message.")
        return None

    def generate_query(self, task: str, data_source: str, schema_or_mapping: Dict[str, Any]) -> Optional[
        Union[SQLQuery, ElasticsearchQuery]]:
        if data_source.lower() == "mysql":
            return self.generate_sql_query(task, schema_or_mapping)
        elif data_source.lower() == "elasticsearch":
            return self.generate_elasticsearch_query(task, schema_or_mapping)
        else:
            self.logger.error(f"Unsupported data source: {data_source}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.2
    )
    query_generator = QueryGenerator(llm)

    # Test SQL query generation
    mysql_schema = {
        "sales": ["date:dat", "product_id:int*", "customer_id:int", "quantity:int", "revenue:dec"],
        "products": ["product_id:int*", "name:var", "category:var", "price:dec"],
    }
    sql_task = "Find the top 5 products by revenue in Q2 of the current year"
    sql_query = query_generator.generate_query(sql_task, "mysql", mysql_schema)
    if sql_query:
        print("Generated SQL Query:")
        print(sql_query.query)
        print("Explanation:")
        print(sql_query.explanation)

    # Test Elasticsearch query generation
    es_mapping = {
        "properties": {
            "order_id": {"type": "keyword"},
            "date": {"type": "date"},
            "customer": {
                "type": "nested",
                "properties": {
                    "id": {"type": "keyword"},
                    "name": {"type": "text"}
                }
            },
            "items": {
                "type": "nested",
                "properties": {
                    "product_id": {"type": "keyword"},
                    "name": {"type": "text"},
                    "quantity": {"type": "integer"},
                    "price": {"type": "float"}
                }
            },
            "total_amount": {"type": "float"}
        }
    }
    es_task = "Find the top 5 customers by total order amount in the last 30 days"
    es_query = query_generator.generate_query(es_task, "elasticsearch", es_mapping)
    if es_query:
        print("\nGenerated Elasticsearch Query:")
        print(es_query.query)
        print("Explanation:")
        print(es_query.explanation)
