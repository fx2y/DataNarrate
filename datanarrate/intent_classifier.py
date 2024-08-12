import logging
import os
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class IntentClassification(BaseModel):
    intents: List[str] = Field(description="The classified intents of the user's query")
    confidences: List[float] = Field(description="Confidence scores for each intent (0-1)")
    explanation: str = Field(description="Brief explanation of why these intents were chosen")


class IntentClassifier:
    def __init__(self, llm: ChatOpenAI, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=IntentClassification)
        self.intents = [
            "data_retrieval", "data_analysis", "visualization", "comparison",
            "trend_analysis", "prediction", "explanation", "summary",
            "anomaly_detection", "correlation_analysis", "what_if_analysis",
            "root_cause_analysis", "recommendation"
        ]
        self.classification_chain = self._create_classification_chain()

    def _create_classification_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify the user's intents based on their query. "
                       f"Possible intents: {', '.join(self.intents)}. "
                       "A query may have multiple intents. List all relevant intents, "
                       "provide a confidence score for each intent, and a brief explanation. "
                       "Classification format: {format_instructions}"),
            ("human", "User query: {query}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def classify(self, query: str) -> Optional[IntentClassification]:
        try:
            self.logger.info(f"Classifying intents for query: {query}")
            classification = self.classification_chain.invoke({"query": query})
            self.logger.info(f"Intents classified as: {classification.intents}")
            return classification
        except Exception as e:
            self.logger.error(f"Error classifying intents: {e}", exc_info=True)
            return None

    def batch_classify(self, queries: List[str]) -> List[Optional[IntentClassification]]:
        self.logger.info(f"Batch classifying {len(queries)} queries")
        results = []
        for query in queries:
            result = self.classify(query)
            results.append(result)
        self.logger.info(
            f"Batch classification completed. Successful: {sum(1 for r in results if r is not None)}, Failed: {sum(1 for r in results if r is None)}")
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = ChatOpenAI(model_name="deepseek-chat", openai_api_base='https://api.deepseek.com',
                     openai_api_key=os.environ["DEEPSEEK_API_KEY"])
    classifier = IntentClassifier(llm)

    test_queries = [
        "Show me the sales data for Q2",
        "What's the trend of customer acquisition over the last 6 months?",
        "Compare the performance of our top 5 products",
        "Predict our revenue for the next quarter based on current data",
        "Explain why we saw a dip in user engagement last month"
    ]

    for query in test_queries:
        result = classifier.classify(query)
        if result:
            print(f"Query: {query}")
            print(f"Intents: {result.intents}")
            print(f"Confidences: {result.confidences}")
            print(f"Explanation: {result.explanation}")
            print("---")
        else:
            print(f"Failed to classify query: {query}")
            print("---")
