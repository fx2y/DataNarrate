import logging
from typing import Dict, Any, Optional, List

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config
from visualization_generator import VisualizationSpec


class OutputFormat(BaseModel):
    summary: str = Field(description="A concise summary of the analysis results")
    key_points: List[str] = Field(description="List of key points from the analysis")
    narrative: str = Field(description="A narrative explanation of the results")
    next_steps: Optional[List[str]] = Field(description="Suggested next steps or follow-up questions", default=None)
    visualizations: Optional[List[VisualizationSpec]] = Field(description="List of visualization specifications",
                                                              default=None)


class OutputGenerator:
    def __init__(self, llm: BaseChatModel, logger: Optional[logging.Logger] = None, **kwargs):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=OutputFormat)
        self.generation_chain = self._create_generation_chain()

    def _create_generation_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an advanced output generator for a data analysis agent. "
                       "Your task is to create clear, concise, and informative summaries "
                       "of analysis results. Adapt your language and detail level to the "
                       "user's expertise and the complexity of the data. "
                       "Incorporate visualization specifications when provided. "
                       "Output format: {format_instructions}"),
            ("human", "Context: {context}\nAnalysis results: {results}\n"
                      "User expertise: {user_expertise}\nUser preferences: {user_preferences}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def generate_output(self, context: str, results: Dict[str, Any], user_expertise: str = "general",
                        user_preferences: Dict[str, Any] = {}) -> OutputFormat:
        try:
            self.logger.info("Generating output for analysis results")

            # Extract visualization specifications from results
            visualizations = []
            for step_result in results.values():
                if isinstance(step_result, VisualizationSpec):
                    visualizations.append(step_result)
                elif isinstance(step_result, dict) and 'output' in step_result and isinstance(step_result['output'],
                                                                                              VisualizationSpec):
                    visualizations.append(step_result['output'])

            output = self.generation_chain.invoke({
                "context": context,
                "results": results,
                "user_expertise": user_expertise,
                "user_preferences": user_preferences
            })

            # Add extracted visualization specifications to the output
            output.visualizations = visualizations

            self.logger.info("Output generated successfully")
            return output
        except Exception as e:
            self.logger.error(f"Error generating output: {e}", exc_info=True)
            return self._generate_error_output()

    def _generate_error_output(self) -> OutputFormat:
        return OutputFormat(
            summary="Error occurred while generating output.",
            key_points=["Unable to process results"],
            narrative="An error occurred during the output generation process. Please try again or contact support if the issue persists.",
            next_steps=["Review the input data", "Try simplifying the query", "Contact support for assistance"]
        )

    def format_output(self, output: OutputFormat, format_type: str = "text") -> str:
        format_functions = {
            "text": self._format_as_text,
            "html": self._format_as_html,
            "markdown": self._format_as_markdown
        }
        formatter = format_functions.get(format_type, self._format_as_text)
        return formatter(output)

    def _format_as_text(self, output: OutputFormat) -> str:
        formatted = f"Summary: {output.summary}\n\n"
        formatted += "Key Points:\n" + "\n".join(f"- {point}" for point in output.key_points) + "\n\n"
        formatted += f"Narrative: {output.narrative}\n\n"
        if output.next_steps:
            formatted += "Next Steps:\n" + "\n".join(f"- {step}" for step in output.next_steps) + "\n\n"
        if output.visualizations:
            formatted += "Visualizations:\n"
            for i, viz in enumerate(output.visualizations, 1):
                formatted += f"Visualization {i}:\n"
                formatted += f"  Type: {viz.chart_type}\n"
                formatted += f"  Title: {viz.title}\n"
                if viz.x_axis:
                    formatted += f"  X-axis: {viz.x_axis}\n"
                if viz.y_axis:
                    formatted += f"  Y-axis: {viz.y_axis}\n"
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
        if output.visualizations:
            html += "<h2>Visualizations</h2>"
            for i, viz in enumerate(output.visualizations, 1):
                html += f"<h3>Visualization {i}</h3>"
                html += f"<p>Type: {viz.chart_type}</p>"
                html += f"<p>Title: {viz.title}</p>"
                if viz.x_axis:
                    html += f"<p>X-axis: {viz.x_axis}</p>"
                if viz.y_axis:
                    html += f"<p>Y-axis: {viz.y_axis}</p>"
        return html

    def _format_as_markdown(self, output: OutputFormat) -> str:
        markdown = f"## Summary\n\n{output.summary}\n\n"
        markdown += "## Key Points\n\n" + "\n".join(f"- {point}" for point in output.key_points) + "\n\n"
        markdown += f"## Narrative\n\n{output.narrative}\n\n"
        if output.next_steps:
            markdown += "## Next Steps\n\n" + "\n".join(f"- {step}" for step in output.next_steps) + "\n\n"
        if output.visualizations:
            markdown += "## Visualizations\n\n"
            for i, viz in enumerate(output.visualizations, 1):
                markdown += f"### Visualization {i}\n\n"
                markdown += f"- Type: {viz.chart_type}\n"
                markdown += f"- Title: {viz.title}\n"
                if viz.x_axis:
                    markdown += f"- X-axis: {viz.x_axis}\n"
                if viz.y_axis:
                    markdown += f"- Y-axis: {viz.y_axis}\n"
                markdown += "\n"
        return markdown


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = ChatOpenAI(model_name=config.LLM_MODEL_NAME, openai_api_base=config.OPENAI_API_BASE,
                     openai_api_key=config.OPENAI_API_KEY, temperature=0.2)
    generator = OutputGenerator(llm)

    # Example usage
    context = "Analyzing Q2 sales data for top-performing products"
    results = {
        "step_1": {
            "total_sales": 1000000,
            "top_products": ["Product A", "Product B", "Product C"],
            "growth_rate": 0.15,
            "regional_performance": {"North": 0.3, "South": 0.2, "East": 0.25, "West": 0.25}
        },
        "step_2": VisualizationSpec(
            chart_type="bar",
            title="Top 5 Products by Revenue in Q2",
            x_axis="Product",
            y_axis="Revenue",
            data_series=[{"Product A": 300000}, {"Product B": 250000}, {"Product C": 200000}, {"Product D": 150000},
                         {"Product E": 100000}],
            color_scheme=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        )
    }
    user_preferences = {"chart_type": "bar", "color_scheme": "blue"}

    output = generator.generate_output(context, results, user_expertise="business analyst",
                                       user_preferences=user_preferences)
    print("Text Output:")
    print(generator.format_output(output, format_type="text"))
    print("\nHTML Output:")
    print(generator.format_output(output, format_type="html"))
    print("\nMarkdown Output:")
    print(generator.format_output(output, format_type="markdown"))
