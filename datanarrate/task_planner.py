import json
import logging
import os
from typing import List, Dict, Any

from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from query_analyzer import QueryAnalysis


class TaskStep(BaseModel):
    step_number: int = Field(description="The order of the step in the plan")
    description: str = Field(description="A clear, concise description of the step")
    required_capability: str = Field(description="The high-level capability required for this step")
    input_description: Dict[str, str] = Field(description="Description of required inputs for this step", default={})
    tools: List[str] = Field(description="List of tools that might be useful for this step", default=[])


class TaskPlan(BaseModel):
    steps: List[TaskStep] = Field(description="The list of steps in the task plan")
    reasoning: str = Field(description="Explanation of the planning process")


class TaskPlanner:
    def __init__(self, llm: BaseLLM, logger: logging.Logger = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=TaskPlan)
        self.plan_chain = self._create_plan_chain()

    def _create_plan_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a task planner for a data analysis system. "
                       "Given a query analysis, create a detailed plan to accomplish the task. "
                       "Consider the task type, sub-tasks, relevant intents, and potential tools. "
                       "Ensure each step is clear and actionable. "
                       "Output format: {format_instructions}"),
            ("human", "Query Analysis: {query_analysis}\nContext: {context}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def plan_task(self, query_analysis: QueryAnalysis, context: Dict[str, Any]) -> TaskPlan:
        try:
            self.logger.info("Planning task based on query analysis")
            plan = self.plan_chain.invoke({
                "query_analysis": json.dumps(query_analysis.dict(), default=str),  # Use .dict() and json.dumps()
                "context": str(context)
            })
            self.logger.info(f"Task plan created: {plan}")
            return plan
        except Exception as e:
            self.logger.error(f"Error planning task: {e}", exc_info=True)
            return None

    def replan(self, previous_plan: TaskPlan, feedback: str, context: Dict[str, Any]) -> TaskPlan:
        try:
            self.logger.info("Replanning task based on feedback")
            replan_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a task planner for a data analysis system. "
                           "Given a previous plan and feedback, create an updated plan. "
                           "Consider the feedback carefully and adjust the plan accordingly. "
                           "Output format: {format_instructions}"),
                ("human", "Previous Plan: {previous_plan}\nFeedback: {feedback}\nContext: {context}")
            ]).partial(format_instructions=self.output_parser.get_format_instructions())

            replan_chain = replan_prompt | self.llm | self.output_parser
            updated_plan = replan_chain.invoke({
                "previous_plan": json.dumps(previous_plan.dict(), default=str),  # Use .dict() and json.dumps()
                "feedback": feedback,
                "context": str(context)
            })
            self.logger.info(f"Updated task plan created: {updated_plan}")
            return updated_plan
        except Exception as e:
            self.logger.error(f"Error replanning task: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from query_analyzer import QueryAnalyzer

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize LLM
    llm = ChatOpenAI(model_name="deepseek-chat", openai_api_base='https://api.deepseek.com',
                     openai_api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0.2)

    # Initialize QueryAnalyzer and TaskPlanner
    query_analyzer = QueryAnalyzer(llm, logger)
    task_planner = TaskPlanner(llm, logger)

    # Test the TaskPlanner
    test_query = "Show me a bar chart of our top 5 selling products in Q2, including their revenue and compare it with last year's Q2 performance"
    test_intents = ["data_visualization", "sales_analysis", "time_comparison"]

    query_analysis = query_analyzer.analyze_query(test_query, test_intents)
    context = {"user_role": "data analyst", "data_access_level": "full"}

    if query_analysis:
        task_plan = task_planner.plan_task(query_analysis, context)
        if task_plan:
            print("Initial Task Plan:")
            for step in task_plan.steps:
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Required Capability: {step.required_capability}")
                print(f"  Tools: {', '.join(step.tools)}")
            print(f"\nReasoning: {task_plan.reasoning}")

            # Test replanning
            feedback = "The plan looks good, but we need to add a step to check for any data anomalies before visualization."
            updated_plan = task_planner.replan(task_plan, feedback, context)
            if updated_plan:
                print("\nUpdated Task Plan:")
                for step in updated_plan.steps:
                    print(f"Step {step.step_number}: {step.description}")
                    print(f"  Required Capability: {step.required_capability}")
                    print(f"  Tools: {', '.join(step.tools)}")
                print(f"\nReasoning: {updated_plan.reasoning}")
        else:
            print("Task planning failed.")
    else:
        print("Query analysis failed.")
