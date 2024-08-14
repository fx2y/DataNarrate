import logging
from typing import Dict, Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config


class InsightPoint(BaseModel):
    topic: str = Field(description="The main topic or subject of the insight")
    description: str = Field(description="A detailed description of the insight")
    relevance: float = Field(description="A score from 0 to 1 indicating the relevance of this insight", ge=0, le=1)
    data_points: List[str] = Field(description="Key data points supporting this insight")


class Storyline(BaseModel):
    title: str = Field(description="An engaging title for the data story")
    summary: str = Field(description="A brief summary of the main findings")
    key_insights: List[InsightPoint] = Field(description="A list of key insights derived from the data")
    narrative: str = Field(description="A cohesive narrative tying together the insights")
    next_steps: List[str] = Field(description="Suggested next steps or areas for further investigation")


class StorylineCreator:
    def __init__(self, llm: BaseChatModel, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=Storyline)
        self.generation_chain = self._create_generation_chain()

    def _create_generation_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data storyteller. Given a set of analysis results, "
                       "create a compelling narrative that highlights key insights and tells "
                       "a coherent story about the data. Focus on the most important and "
                       "actionable findings. Ensure the narrative is clear, engaging, and "
                       "accessible to the target audience. "
                       "Output format: {format_instructions}"),
            ("human", "Analysis Results: {results}\nContext: {context}\nAudience: {audience}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def create_storyline(self, results: Dict[str, Any], context: str, audience: str) -> Optional[Storyline]:
        try:
            self.logger.info("Generating data storyline")
            self.logger.debug(f"Analysis Results: {results}")
            self.logger.debug(f"Context: {context}")
            self.logger.debug(f"Audience: {audience}")

            storyline = self.generation_chain.invoke({
                "results": results,
                "context": context,
                "audience": audience
            })

            self.logger.info("Storyline generated successfully")
            self.logger.debug(f"Generated Storyline: {storyline}")
            return storyline
        except Exception as e:
            self.logger.error(f"Error generating storyline: {e}", exc_info=True)
            return None

    def validate_storyline(self, storyline: Storyline) -> bool:
        """
        Validate the generated storyline for completeness and coherence.
        """
        try:
            self.logger.info("Validating generated storyline")

            if not storyline.title or not storyline.summary or not storyline.key_insights or not storyline.narrative:
                self.logger.warning("Invalid storyline: missing required components")
                return False

            if len(storyline.key_insights) < 2:
                self.logger.warning("Invalid storyline: insufficient key insights")
                return False

            for insight in storyline.key_insights:
                if insight.relevance < 0.5:
                    self.logger.warning(f"Low relevance insight detected: {insight.topic}")

            self.logger.info("Storyline validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating storyline: {e}", exc_info=True)
            return False


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

    # Initialize StorylineCreator
    storyline_creator = StorylineCreator(llm)

    # Test data
    test_results = {
        "sales_data": {
            "total_revenue": 1000000,
            "top_products": [
                {"name": "Product A", "revenue": 250000},
                {"name": "Product B", "revenue": 200000},
                {"name": "Product C", "revenue": 150000}
            ],
            "year_over_year_growth": 0.15
        },
        "customer_data": {
            "total_customers": 5000,
            "new_customers": 1000,
            "customer_retention_rate": 0.85
        },
        "market_analysis": {
            "market_share": 0.25,
            "competitor_growth": 0.1
        }
    }

    test_context = "Q2 2023 Performance Review"
    test_audience = "Executive Leadership Team"

    # Generate storyline
    storyline = storyline_creator.create_storyline(test_results, test_context, test_audience)

    if storyline and storyline_creator.validate_storyline(storyline):
        print("Generated Storyline:")
        print(f"Title: {storyline.title}")
        print(f"Summary: {storyline.summary}")
        print("Key Insights:")
        for insight in storyline.key_insights:
            print(f"- {insight.topic} (Relevance: {insight.relevance})")
            print(f"  Description: {insight.description}")
            print(f"  Data Points: {', '.join(insight.data_points)}")
        print(f"Narrative: {storyline.narrative}")
        print("Next Steps:")
        for step in storyline.next_steps:
            print(f"- {step}")
    else:
        print("Failed to generate a valid storyline")
