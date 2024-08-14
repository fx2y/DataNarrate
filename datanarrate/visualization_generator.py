import logging
from typing import Dict, Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config


class VisualizationSpec(BaseModel):
    chart_type: str = Field(description="Type of chart (e.g., bar, line, pie, scatter)")
    title: str = Field(description="Title of the visualization")
    x_axis: Optional[str] = Field(description="X-axis label", default=None)
    y_axis: Optional[str] = Field(description="Y-axis label", default=None)
    data_series: List[Dict[str, Any]] = Field(description="Data series for the visualization")
    color_scheme: Optional[List[str]] = Field(description="Color scheme for the visualization", default=None)
    additional_options: Optional[Dict[str, Any]] = Field(description="Additional chart-specific options", default=None)


class VisualizationGenerator:
    def __init__(self, llm: BaseChatModel, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=VisualizationSpec)
        self.generation_chain = self._create_generation_chain()

    def _create_generation_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data visualization specialist. "
                       "Given a dataset and visualization requirements, create a detailed "
                       "visualization specification. Consider the data types, relationships, "
                       "and the best way to represent the information visually. "
                       "Output format: {format_instructions}"),
            ("human", "Data: {data}\nRequirements: {requirements}\nUser Preferences: {user_preferences}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def generate_visualization(self, data: Dict[str, Any], requirements: str,
                               user_preferences: Optional[Dict[str, Any]] = None) -> Optional[VisualizationSpec]:
        try:
            self.logger.info("Generating visualization specification")
            self.logger.debug(f"Data: {data}")
            self.logger.debug(f"Requirements: {requirements}")
            self.logger.debug(f"User Preferences: {user_preferences}")

            visualization_spec = self.generation_chain.invoke({
                "data": data,
                "requirements": requirements,
                "user_preferences": user_preferences or {}
            })

            self.logger.info(f"Generated visualization specification: {visualization_spec}")
            return visualization_spec
        except Exception as e:
            self.logger.error(f"Error generating visualization specification: {e}", exc_info=True)
            return None

    def validate_spec(self, spec: VisualizationSpec) -> bool:
        """
        Validate the generated visualization specification.
        """
        try:
            self.logger.info("Validating visualization specification")
            # Add validation logic here
            # For example, check if the chart type is supported, if the data series match the chart type, etc.
            # For now, we'll just do a basic check
            if not spec.chart_type or not spec.data_series:
                self.logger.warning("Invalid visualization specification: missing chart type or data series")
                return False
            self.logger.info("Visualization specification is valid")
            return True
        except Exception as e:
            self.logger.error(f"Error validating visualization specification: {e}", exc_info=True)
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

    # Initialize VisualizationGenerator
    viz_generator = VisualizationGenerator(llm)

    # Test data
    test_data = {
        "product_sales": [
            {"product": "Product A", "sales_2023_q2": 1000, "sales_2022_q2": 800},
            {"product": "Product B", "sales_2023_q2": 1200, "sales_2022_q2": 1000},
            {"product": "Product C", "sales_2023_q2": 800, "sales_2022_q2": 900},
            {"product": "Product D", "sales_2023_q2": 1500, "sales_2022_q2": 1300},
            {"product": "Product E", "sales_2023_q2": 900, "sales_2022_q2": 750},
        ]
    }

    # Test requirements
    test_requirements = "Create a bar chart comparing the top 5 selling products in Q2 2023 with their performance in Q2 2022"

    # Test user preferences
    test_preferences = {
        "color_scheme": ["#1f77b4", "#ff7f0e"],
        "chart_style": "modern"
    }

    # Generate visualization specification
    viz_spec = viz_generator.generate_visualization(test_data, test_requirements, test_preferences)

    if viz_spec and viz_generator.validate_spec(viz_spec):
        print("Generated Visualization Specification:")
        print(f"Chart Type: {viz_spec.chart_type}")
        print(f"Title: {viz_spec.title}")
        print(f"X-axis: {viz_spec.x_axis}")
        print(f"Y-axis: {viz_spec.y_axis}")
        print(f"Data Series: {viz_spec.data_series}")
        print(f"Color Scheme: {viz_spec.color_scheme}")
        print(f"Additional Options: {viz_spec.additional_options}")
    else:
        print("Failed to generate a valid visualization specification")
