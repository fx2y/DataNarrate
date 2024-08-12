import logging
import os
from typing import Dict, Any, List, Optional

from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


class QueryAnalysis(BaseModel):
    task_type: str = Field(description="The type of task: data_retrieval, visualization, storytelling, or custom")
    sub_task_types: List[str] = Field(
        description="Specific sub-tasks identified (e.g., trend_analysis, comparison, forecasting)")
    confidence: float = Field(description="Confidence score for the task type classification")
    relevant_intents: List[str] = Field(description="List of relevant intents from the intent classifier")
    action_heuristics: Dict[str, float] = Field(description="Heuristic scores for different actions")
    required_data_sources: List[str] = Field(description="List of data sources likely needed for this query")
    potential_tools: List[str] = Field(description="List of tools that might be useful for this query")
    missing_information: List[str] = Field(description="Information that seems to be missing from the query")


class QueryAnalyzer:
    def __init__(self, llm: BaseLLM, logger: logging.Logger = None, schema_info: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.schema_info = schema_info or {}
        self.output_parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
        self.analyze_chain = self._create_analyze_chain()

    def _create_analyze_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the given query considering the following aspects:\n"
                       "1. Classify the main task type and identify sub-tasks.\n"
                       "2. Consider the given intents and schema information.\n"
                       "3. Suggest potential data sources and tools needed.\n"
                       "4. Identify any missing information in the query.\n"
                       "Output format: {format_instructions}"),
            ("human", "Query: {query}\nIntents: {intents}\nSchema Info: {schema_info}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def analyze_query(self, query: str, intents: List[str]) -> QueryAnalysis:
        try:
            self.logger.info(f"Analyzing query: {query}")
            analysis = self.analyze_chain.invoke({
                "query": query,
                "intents": ", ".join(intents),
                "schema_info": str(self.schema_info)
            })
            self.logger.info(f"Query analysis completed: {analysis}")
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing query: {e}", exc_info=True)
            return None

    def update_schema_info(self, new_schema_info: Dict[str, Any]):
        self.schema_info.update(new_schema_info)
        self.logger.info("Schema information updated")


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize LLM
    llm = ChatOpenAI(model_name="deepseek-chat", openai_api_base='https://api.deepseek.com',
                     openai_api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0.2)

    # Sample schema info (this would typically come from the Schema Retriever)
    sample_schema = {
        "tables": ["sales", "products", "customers"],
        "sales": ["date", "product_id", "customer_id", "quantity", "revenue"],
        "products": ["product_id", "name", "category", "price"],
        "customers": ["customer_id", "name", "location"]
    }

    # Initialize QueryAnalyzer
    analyzer = QueryAnalyzer(llm, logger, schema_info=sample_schema)

    # Test the QueryAnalyzer
    test_query = "Show me a bar chart of our top 5 selling products in Q2, including their revenue and compare it with last year's Q2 performance"
    test_intents = ["data_visualization", "sales_analysis", "time_comparison"]

    result = analyzer.analyze_query(test_query, test_intents)
    if result:
        print(f"Task Type: {result.task_type}")
        print(f"Sub-task Types: {result.sub_task_types}")
        print(f"Confidence: {result.confidence}")
        print(f"Relevant Intents: {result.relevant_intents}")
        print(f"Action Heuristics: {result.action_heuristics}")
        print(f"Required Data Sources: {result.required_data_sources}")
        print(f"Potential Tools: {result.potential_tools}")
        print(f"Missing Information: {result.missing_information}")
    else:
        print("Query analysis failed.")

    # Simulating an update to schema info (e.g., when a new data source becomes available)
    new_schema_info = {"new_table": ["column1", "column2"]}
    analyzer.update_schema_info(new_schema_info)
