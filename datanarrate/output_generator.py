import logging
import os
from typing import Dict, Any, Optional, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class OutputFormat(BaseModel):
    summary: str = Field(description="A concise summary of the analysis results")
    key_points: List[str] = Field(description="List of key points from the analysis")
    narrative: str = Field(description="A narrative explanation of the results")
    next_steps: Optional[List[str]] = Field(description="Suggested next steps or follow-up questions", default=None)


class OutputGenerator:
    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None, **kwargs):
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        self.llm = self._create_llm(model_name, **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=OutputFormat)
        self.generation_chain = self._create_generation_chain()

    def _create_llm(self, model_name: str, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(model_name=model_name, temperature=0.2, **kwargs)

    def _create_generation_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an advanced output generator for a data analysis agent. "
                       "Your task is to create clear, concise, and informative summaries "
                       "of analysis results. Adapt your language and detail level to the "
                       "user's expertise and the complexity of the data. "
                       "Output format: {format_instructions}"),
            ("human", "Context: {context}\nAnalysis results: {results}\nUser expertise: {user_expertise}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def generate_output(self, context: str, results: Dict[str, Any], user_expertise: str = "general") -> OutputFormat:
        try:
            self.logger.info("Generating output for analysis results")
            output = self.generation_chain.invoke({
                "context": context,
                "results": results,
                "user_expertise": user_expertise
            })
            self.logger.info("Output generated successfully")
            return output
        except Exception as e:
            self.logger.error(f"Error generating output: {e}", exc_info=True)
            return OutputFormat(
                summary="Error occurred while generating output.",
                key_points=["Unable to process results"],
                narrative="An error occurred during the output generation process. Please try again or contact support if the issue persists.",
                next_steps=["Review the input data", "Try simplifying the query", "Contact support for assistance"]
            )

    def format_output(self, output: OutputFormat, format_type: str = "text") -> str:
        if format_type == "text":
            return self._format_as_text(output)
        elif format_type == "html":
            return self._format_as_html(output)
        else:
            self.logger.warning(f"Unsupported format type: {format_type}. Defaulting to text.")
            return self._format_as_text(output)

    def _format_as_text(self, output: OutputFormat) -> str:
        formatted = f"Summary: {output.summary}\n\n"
        formatted += "Key Points:\n" + "\n".join(f"- {point}" for point in output.key_points) + "\n\n"
        formatted += f"Narrative: {output.narrative}\n\n"
        if output.next_steps:
            formatted += "Next Steps:\n" + "\n".join(f"- {step}" for step in output.next_steps)
        return formatted

    def _format_as_html(self, output: OutputFormat) -> str:
        html = f"<h2>Summary</h2><p>{output.summary}</p>"
        html += "<h2>Key Points</h2><ul>"
        for point in output.key_points:
            html += f"<li>{point}</li>"
        html += "</ul>"
        html += f"<h2>Narrative</h2><p>{output.narrative}</p>"
        if output.next_steps:
            html += "<h2>Next Steps</h2><ul>"
            for step in output.next_steps:
                html += f"<li>{step}</li>"
            html += "</ul>"
        return html


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generator = OutputGenerator("deepseek-chat", openai_api_base='https://api.deepseek.com',
                                openai_api_key=os.environ["DEEPSEEK_API_KEY"])

    # Example usage
    context = "Analyzing Q2 sales data for top-performing products"
    results = {
        "total_sales": 1000000,
        "top_products": ["Product A", "Product B", "Product C"],
        "growth_rate": 0.15,
        "regional_performance": {"North": 0.3, "South": 0.2, "East": 0.25, "West": 0.25}
    }

    output = generator.generate_output(context, results, user_expertise="business analyst")
    print(generator.format_output(output, format_type="text"))
    print("\nHTML Output:")
    print(generator.format_output(output, format_type="html"))
