import logging
import os
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class IntentClassification(BaseModel):
    intent: str = Field(description="The classified intent of the user's query")
    confidence: float = Field(description="Confidence score of the classification (0-1)")
    explanation: str = Field(description="Brief explanation of why this intent was chosen")


class IntentClassifier:
    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None, **kwargs):
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        self.llm = self._create_llm(model_name, **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=IntentClassification)
        self.classification_chain = self._create_classification_chain()

    def _create_llm(self, model_name: str, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(model_name=model_name, temperature=0, **kwargs)

    def _create_classification_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify the user's intent based on their query. "
                       "Possible intents: data_retrieval, data_analysis, visualization, comparison, trend_analysis, prediction, explanation, summary. "
                       "Respond with the intent, confidence score, and a brief explanation. "
                       "Classification format: {format_instructions}"),
            ("human", "User query: {query}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def classify(self, query: str) -> Optional[IntentClassification]:
        try:
            self.logger.info(f"Classifying intent for query: {query}")
            classification = self.classification_chain.invoke({"query": query})
            self.logger.info(f"Intent classified as: {classification.intent}")
            return classification
        except Exception as e:
            self.logger.error(f"Error classifying intent: {e}", exc_info=True)
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
    classifier = IntentClassifier("deepseek-chat", openai_api_base='https://api.deepseek.com',
                                  openai_api_key=os.environ["DEEPSEEK_API_KEY"])
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
            print(f"Intent: {result.intent}")
            print(f"Confidence: {result.confidence}")
            print(f"Explanation: {result.explanation}")
            print("---")
        else:
            print(f"Failed to classify query: {query}")
            print("---")
