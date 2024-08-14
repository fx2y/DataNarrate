import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import config


class ReasoningOutput(BaseModel):
    evaluation: str = Field(description="Evaluation of the current results")
    decision: str = Field(description="Decision on the next action to take")
    explanation: str = Field(description="Explanation of the reasoning process")
    confidence: float = Field(description="Confidence level in the decision (0-1)")
    result_quality: float = Field(description="Assessment of the quality/completeness of current results (0-1)")


class ReasoningStrategy(BaseModel):
    name: str = Field(description="Name of the reasoning strategy")
    description: str = Field(description="Description of when to use this strategy")
    prompt_template: str = Field(description="Prompt template for this strategy")


class ReasoningEngine:
    def __init__(self, llm: BaseChatModel, logger: Optional[logging.Logger] = None, **kwargs):
        self.llm = llm
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.output_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=ReasoningOutput)
        self.strategies: Dict[str, ReasoningStrategy] = self._initialize_strategies()
        self.feedback_history: List[Tuple[ReasoningOutput, float]] = []

    def _initialize_strategies(self) -> Dict[str, ReasoningStrategy]:
        return {
            "default": ReasoningStrategy(
                name="default",
                description="General-purpose reasoning strategy",
                prompt_template=(
                    "You are an advanced reasoning engine for a data analysis agent. "
                    "Evaluate the current results, make decisions on next steps, "
                    "and provide explanations for your reasoning process. "
                    "Consider the overall task, available tools, and potential risks. "
                    "Output format: {format_instructions}\n"
                    "Task: {task}\n"
                    "Current results: {results}\n"
                    "Available tools: {tools}\n"
                    "Historical context: {history}\n"
                    "Provide your reasoning and decision on the next step."
                )
            ),
            "complex": ReasoningStrategy(
                name="complex",
                description="Strategy for complex tasks requiring in-depth analysis",
                prompt_template=(
                    "You are an expert data analyst tackling a complex problem. "
                    "Break down the problem, consider multiple approaches, "
                    "and provide a detailed reasoning process. "
                    "Output format: {format_instructions}\n"
                    "Task: {task}\n"
                    "Current results: {results}\n"
                    "Available tools: {tools}\n"
                    "Historical context: {history}\n"
                    "Provide a comprehensive analysis and decision on the next steps."
                )
            )
        }

    def _create_reasoning_chain(self, strategy: str) -> Any:
        strategy_obj = self.strategies.get(strategy, self.strategies["default"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", strategy_obj.prompt_template),
            ("human", "Provide your reasoning and decision based on the given information. "
                      "Also, assess the quality and completeness of the current results on a scale of 0 to 1, "
                      "where 0 means the results are completely inadequate or missing, and 1 means the results are perfect and complete.")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def reason(self, task: str, results: Dict[str, Any], available_tools: List[str],
               strategy: str = "default", history: Optional[List[str]] = None) -> ReasoningOutput:
        try:
            self.logger.info(f"Reasoning about task: {task} using strategy: {strategy}")
            reasoning_chain = self._create_reasoning_chain(strategy)
            reasoning_output = reasoning_chain.invoke({
                "task": task,
                "results": json.dumps(results, indent=2),
                "tools": ", ".join(available_tools),
                "history": "\n".join(history or [])
            })
            self.logger.info(f"Reasoning complete. Decision: {reasoning_output.decision}")
            return reasoning_output
        except Exception as e:
            self.logger.error(f"Error in reasoning process: {e}", exc_info=True)
            return ReasoningOutput(
                evaluation="Error occurred during reasoning",
                decision="Fallback to default action or request human intervention",
                explanation=f"An error occurred: {str(e)}",
                confidence=0.0,
                result_quality=0.0
            )

    def chain_of_thought(self, task: str, intermediate_results: List[Dict[str, Any]],
                         available_tools: List[str], strategy: str = "default") -> List[ReasoningOutput]:
        chain_output = []
        history = []
        for i, result in enumerate(intermediate_results):
            self.logger.info(f"Performing chain-of-thought reasoning step {i + 1}")
            step_output = self.reason(task, result, available_tools, strategy, history)
            chain_output.append(step_output)
            history.append(f"Step {i + 1}: {step_output.decision}")
            if step_output.confidence < 0.5:
                self.logger.warning(f"Low confidence in step {i + 1}. Consider reviewing or getting human input.")
        return chain_output

    def provide_feedback(self, reasoning_output: ReasoningOutput, feedback_score: float) -> None:
        self.feedback_history.append((reasoning_output, feedback_score))
        self.logger.info(f"Feedback received. Score: {feedback_score}")
        if len(self.feedback_history) % 10 == 0:
            self._analyze_feedback()

    def _analyze_feedback(self) -> None:
        # Implement logic to analyze feedback and adjust reasoning strategies
        # This could involve updating prompt templates or adjusting the LLM's parameters
        self.logger.info("Analyzing feedback to improve reasoning strategies")
        # Example: Calculate average feedback score
        avg_score = sum(score for _, score in self.feedback_history) / len(self.feedback_history)
        self.logger.info(f"Average feedback score: {avg_score}")
        # TODO: Implement more sophisticated feedback analysis and strategy adjustment


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.2
    )
    engine = ReasoningEngine(llm)

    # Example usage
    task = "Analyze Q2 sales data and identify top-performing products"
    results = {
        "q2_sales": 1000000,
        "top_products": ["Product A", "Product B", "Product C"],
        "growth_rate": 0.15
    }
    available_tools = ["SQL Query Tool", "Visualization Tool", "Storytelling Tool"]

    reasoning_output = engine.reason(task, results, available_tools, strategy="complex")
    print(f"Evaluation: {reasoning_output.evaluation}")
    print(f"Decision: {reasoning_output.decision}")
    print(f"Explanation: {reasoning_output.explanation}")
    print(f"Confidence: {reasoning_output.confidence}")
    print(f"Result Quality: {reasoning_output.result_quality}")

    # Example of chain-of-thought reasoning
    intermediate_results = [
        {"step1_result": "Retrieved Q2 sales data"},
        {"step2_result": "Identified top 3 products"},
        {"step3_result": "Calculated growth rate"}
    ]
    chain_output = engine.chain_of_thought(task, intermediate_results, available_tools, strategy="complex")
    for i, step in enumerate(chain_output):
        print(f"\nStep {i + 1}:")
        print(f"Evaluation: {step.evaluation}")
        print(f"Decision: {step.decision}")
        print(f"Explanation: {step.explanation}")
        print(f"Confidence: {step.confidence}")
        print(f"Result Quality: {step.result_quality}")

    # Example of providing feedback
    engine.provide_feedback(reasoning_output, 0.8)
