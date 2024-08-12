import json
import logging
from typing import Dict, Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from config import config


class DataSourceInfo(BaseModel):
    name: str = Field(description="Name of the data source (e.g., 'mysql' or 'elasticsearch')")
    relevant_tables_or_indices: List[str] = Field(description="List of relevant tables or indices for this data source")
    suggested_fields: Dict[str, List[str]] = Field(description="Suggested fields for each table or index")


class QueryAnalysis(BaseModel):
    task_type: str = Field(description="The type of task: data_retrieval, visualization, storytelling, or custom")
    sub_task_types: List[str] = Field(
        description="Specific sub-tasks identified (e.g., trend_analysis, comparison, forecasting)")
    relevant_intents: List[str] = Field(description="List of relevant intents from the intent classifier")
    required_data_sources: List[DataSourceInfo] = Field(
        description="List of required data sources with their relevant tables/indices and fields")
    query_constraints: Dict[str, Any] = Field(description="Any constraints or filters identified in the query")
    time_range: Optional[Dict[str, str]] = Field(description="Time range for the query, if applicable")
    suggested_visualizations: Optional[List[str]] = Field(
        description="Suggested visualization types for the query result")
    potential_insights: Optional[List[str]] = Field(
        description="Potential insights that could be derived from the query")


class QueryAnalyzer:
    def __init__(self, llm: BaseChatModel, logger: logging.Logger = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
        self.analyze_chain = self._create_analyze_chain()

    def _create_analyze_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data analyst. Analyze the given query considering the following aspects:\n"
                       "1. Classify the main task type and identify sub-tasks.\n"
                       "2. Consider the given intents and unified schema information.\n"
                       "3. Suggest potential data sources, their relevant tables/indices, and fields needed.\n"
                       "4. Identify query constraints and time ranges if applicable.\n"
                       "5. Suggest appropriate visualizations for the query result.\n"
                       "6. Identify potential insights that could be derived from the query.\n"
                       "7. Ensure that your suggestions for tables/indices and fields exactly match the names provided in the schema.\n"
                       "8. If you're unsure about a table/index or field, refer to the schema information provided.\n"
                       "9. For Elasticsearch nested objects, use dot notation to refer to nested fields.\n\n"
                       "Unified Schema Compression Format Explanation:\n"
                       "- MySQL tables: 'mysql': {{'table_name': ['column:typ?*', ...]}}\n"
                       "  where 'typ' is the first 3 characters of the data type,\n"
                       "  '?' indicates a nullable column, and '*' indicates a primary key.\n"
                       "- Elasticsearch indices: 'elasticsearch': {{'index_name': {{'field': 'typ', ...}}}}\n"
                       "  where 'typ' is the first 3 characters of the field type.\n"
                       "  'nes' indicates a nested object. Nested fields use dot notation.\n"
                       "Be specific and use the actual table/index and field names from the provided schema.\n"
                       "Output format: {format_instructions}"),
            ("human", "Query: {query}\nIntents: {intents}\nUnified Compressed Schema: {schema_info}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def analyze_query(self, query: str, intents: List[str], compressed_schema: Dict[str, Any]) -> Optional[
        QueryAnalysis]:
        try:
            self.logger.info(f"Analyzing query: {query}")
            analysis = self.analyze_chain.invoke({
                "query": query,
                "intents": ", ".join(intents),
                "schema_info": json.dumps(compressed_schema, indent=2)
            })
            self.logger.info(f"Query analysis completed: {analysis}")
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing query: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize LLM
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.2
    )

    # Sample unified compressed schema with nested objects
    sample_schema = {
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

    # Initialize QueryAnalyzer
    analyzer = QueryAnalyzer(llm, logger)

    # Test the QueryAnalyzer with a query involving nested objects
    test_query = "Show me the top 5 customers who have spent the most on electronics products in Q2, including their order details and product specifications"
    test_intents = ["data_visualization", "sales_analysis", "customer_analysis"]

    result = analyzer.analyze_query(test_query, test_intents, sample_schema)
    if result:
        print(f"Task Type: {result.task_type}")
        print(f"Sub-task Types: {result.sub_task_types}")
        print(f"Relevant Intents: {result.relevant_intents}")
        print("Required Data Sources:")
        for ds in result.required_data_sources:
            print(f"  - {ds.name}:")
            print(f"    Tables/Indices: {ds.relevant_tables_or_indices}")
            print(f"    Suggested Fields: {ds.suggested_fields}")
        print(f"Query Constraints: {result.query_constraints}")
        print(f"Time Range: {result.time_range}")
        print(f"Suggested Visualizations: {result.suggested_visualizations}")
        print(f"Potential Insights: {result.potential_insights}")
    else:
        print("Query analysis failed.")
