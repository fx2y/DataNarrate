import json
import logging
import re
from typing import Dict, Any, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config
from datanarrate.task_planner import QueryInfo


class SQLQuery(BaseModel):
    query: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Explanation of the generated query")


class ElasticsearchQuery(BaseModel):
    query: Dict[str, Any] = Field(description="The generated Elasticsearch query")
    explanation: str = Field(description="Explanation of the generated query")


class QueryGenerator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.sql_output_parser = PydanticOutputParser(pydantic_object=SQLQuery)
        self.es_output_parser = PydanticOutputParser(pydantic_object=ElasticsearchQuery)
        self.logger = logging.getLogger(__name__)

    def generate_sql_query(self, task: str, schema: Dict[str, Any], query_info: Optional[QueryInfo] = None) -> Optional[
        SQLQuery]:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert SQL query generator. Generate an SQL query based on the given task, schema, and query information. "
             "Ensure the query is correct and optimized. "
             "Output format: {format_instructions}"),
            ("human", "Task: {task}\nSchema: {schema}\nQuery Info: {query_info}")
        ]).partial(format_instructions=self.sql_output_parser.get_format_instructions())

        try:
            result = self.llm.invoke(prompt.format(
                task=task,
                schema=json.dumps(schema, indent=2),
                query_info=json.dumps(query_info.dict() if query_info else {}, indent=2)
            ))
            self.logger.debug(f"LLM response for SQL query: {result.content}")
            json_str = self._extract_json(result.content)
            return self.sql_output_parser.parse(json_str)
        except Exception as e:
            self.logger.error(f"Error generating SQL query: {e}")
            return None

    def generate_elasticsearch_query(self, task: str, schema: Dict[str, Any], query_info: Optional[QueryInfo] = None) -> \
            Optional[ElasticsearchQuery]:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert Elasticsearch query generator. Generate an Elasticsearch query based on the given task, schema, and query information. "
             "Ensure the query is correct and optimized. "
             "Output format: {format_instructions}"),
            ("human", "Task: {task}\nSchema: {schema}\nQuery Info: {query_info}")
        ]).partial(format_instructions=self.es_output_parser.get_format_instructions())

        try:
            result = self.llm.invoke(prompt.format(
                task=task,
                schema=json.dumps(schema, indent=2),
                query_info=json.dumps(query_info.dict() if query_info else {}, indent=2)
            ))
            self.logger.debug(f"LLM response: {result.content}")

            # Extract the JSON part from the result
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Attempt to parse the JSON string
                try:
                    json_obj = json.loads(json_str)
                    # If successful, create an ElasticsearchQuery object
                    return ElasticsearchQuery(query=json_obj.get('query', {}),
                                              explanation=json_obj.get('explanation', ''))
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON: {e}")
                    return None
            else:
                self.logger.error("No valid JSON found in the output")
                return None
        except Exception as e:
            self.logger.error(f"Error generating Elasticsearch query: {e}")
            return None

    def generate_query(self, task: str, data_source: str, schema_or_mapping: Dict[str, Any],
                       query_info: Optional[QueryInfo] = None) -> Optional[Union[SQLQuery, ElasticsearchQuery]]:
        if data_source.lower() == "mysql":
            return self.generate_sql_query(task, schema_or_mapping, query_info)
        elif data_source.lower() == "elasticsearch":
            return self.generate_elasticsearch_query(task, schema_or_mapping, query_info)
        else:
            self.logger.error(f"Unsupported data source: {data_source}")
            return None

    def _extract_json(self, text: str) -> str:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
        else:
            raise ValueError("No valid JSON found in the output")


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
    else:
        print("Failed to generate SQL query")

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
        print(json.dumps(es_query.query, indent=2))
        print("Explanation:")
        print(es_query.explanation)
    else:
        print("Failed to generate Elasticsearch query")
