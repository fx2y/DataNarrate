import json
import logging
import os
from typing import List, Dict, Any, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class ReasoningOutput(BaseModel):
    evaluation: str = Field(description="Evaluation of the current results")
    decision: str = Field(description="Decision on the next action to take")
    explanation: str = Field(description="Explanation of the reasoning process")
    confidence: float = Field(description="Confidence level in the decision (0-1)")


class ReasoningEngine:
    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None, **kwargs):
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        self.llm = self._create_llm(model_name, **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=ReasoningOutput)
        self.reasoning_chain = self._create_reasoning_chain()

    def _create_llm(self, model_name: str, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(model_name=model_name, temperature=0.2, **kwargs)

    def _create_reasoning_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an advanced reasoning engine for a data analysis agent. "
                       "Evaluate the current results, make decisions on next steps, "
                       "and provide explanations for your reasoning process. "
                       "Consider the overall task, available tools, and potential risks. "
                       "Output format: {format_instructions}"),
            ("human", "Task: {task}\nCurrent results: {results}\nAvailable tools: {tools}\n"
                      "Provide your reasoning and decision on the next step.")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def reason(self, task: str, results: Dict[str, Any], available_tools: List[str]) -> ReasoningOutput:
        try:
            self.logger.info(f"Reasoning about task: {task}")
            reasoning_output = self.reasoning_chain.invoke({
                "task": task,
                "results": json.dumps(results, indent=2),
                "tools": ", ".join(available_tools)
            })
            self.logger.info(f"Reasoning complete. Decision: {reasoning_output.decision}")
            return reasoning_output
        except Exception as e:
            self.logger.error(f"Error in reasoning process: {e}", exc_info=True)
            return ReasoningOutput(
                evaluation="Error occurred during reasoning",
                decision="Fallback to default action or request human intervention",
                explanation=f"An error occurred: {str(e)}",
                confidence=0.0
            )

    def chain_of_thought(self, task: str, intermediate_results: List[Dict[str, Any]], available_tools: List[str]) -> \
            List[ReasoningOutput]:
        chain_output = []
        for i, result in enumerate(intermediate_results):
            self.logger.info(f"Performing chain-of-thought reasoning step {i + 1}")
            step_output = self.reason(task, result, available_tools)
            chain_output.append(step_output)
            if step_output.confidence < 0.5:
                self.logger.warning(f"Low confidence in step {i + 1}. Consider reviewing or getting human input.")
        return chain_output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ReasoningEngine("deepseek-chat", openai_api_base='https://api.deepseek.com',
                             openai_api_key=os.environ["DEEPSEEK_API_KEY"])

    # Example usage
    task = "Analyze Q2 sales data and identify top-performing products"
    results = {
        "q2_sales": 1000000,
        "top_products": ["Product A", "Product B", "Product C"],
        "growth_rate": 0.15
    }
    available_tools = ["SQL Query Tool", "Visualization Tool", "Storytelling Tool"]

    reasoning_output = engine.reason(task, results, available_tools)
    print(f"Evaluation: {reasoning_output.evaluation}")
    print(f"Decision: {reasoning_output.decision}")
    print(f"Explanation: {reasoning_output.explanation}")
    print(f"Confidence: {reasoning_output.confidence}")

    # Example of chain-of-thought reasoning
    intermediate_results = [
        {"step1_result": "Retrieved Q2 sales data"},
        {"step2_result": "Identified top 3 products"},
        {"step3_result": "Calculated growth rate"}
    ]
    chain_output = engine.chain_of_thought(task, intermediate_results, available_tools)
    for i, step in enumerate(chain_output):
        print(f"\nStep {i + 1}:")
        print(f"Evaluation: {step.evaluation}")
        print(f"Decision: {step.decision}")
        print(f"Explanation: {step.explanation}")
        print(f"Confidence: {step.confidence}")
