import json
import logging
import os
from typing import List, Optional, Dict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from intent_classifier import IntentClassifier


class TaskStep(BaseModel):
    step_number: int = Field(description="The order of the step in the plan")
    description: str = Field(description="A clear, concise description of the step")
    required_capability: str = Field(description="The high-level capability required for this step")
    input_description: Dict[str, str] = Field(description="Description of required inputs for this step", default={})


class TaskPlan(BaseModel):
    steps: List[TaskStep] = Field(description="The list of steps in the task plan")
    reasoning: str = Field(description="Explanation of the planning process")


class TaskPlanner:
    def __init__(self, intent_classifier: IntentClassifier, model_name: str = "gpt-4o-mini",
                 logger: Optional[logging.Logger] = None, **kwargs):
        self.intent_classifier = intent_classifier
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        self.llm = self._create_llm(model_name, **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=TaskPlan)
        self.create_plan_chain = self._create_plan_chain()
        self.replan_chain = self._create_replan_chain()

    def _create_llm(self, model_name: str, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(model_name=model_name, temperature=0.2, **kwargs)

    def _create_plan_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Break down the given task into clear, actionable steps. "
                       "Consider the user's intent, current context, and available capabilities. "
                       "For each step, specify the required high-level capability and describe any necessary inputs. "
                       "Do not specify exact tool names, only describe the required capability. "
                       "Provide a plan with steps and reasoning. "
                       "Plan format: {format_instructions}"),
            ("human", "Task: {task}\nIntent: {intent}\nAvailable capabilities: {capabilities}\nContext: {context}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def _create_replan_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Revise the given plan based on the feedback provided. "
                       "Ensure the new plan addresses the feedback while maintaining the overall goal. "
                       "For each step, specify the tool to be used and any necessary input parameters. "
                       "Plan format: {format_instructions}"),
            ("human", "Original plan: {original_plan}\nFeedback: {feedback}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def plan_tasks(self, query: str):
        intent_classification = self.intent_classifier.classify(query)
        # Use the intent_classification to inform task planning
        # ... implementation details ...

    def create_plan(self, task: str, intent: str, capabilities: List[str], context: str) -> Optional[TaskPlan]:
        try:
            self.logger.info(f"Creating plan for task: {task}")
            plan = self.create_plan_chain.invoke({
                "task": task,
                "intent": intent,
                "capabilities": ", ".join(capabilities),
                "context": context
            })
            self.logger.info("Plan created successfully")
            return plan
        except Exception as e:
            self.logger.error(f"Error creating plan: {e}", exc_info=True)
            return None

    def replan(self, original_plan: TaskPlan, feedback: str) -> Optional[TaskPlan]:
        try:
            self.logger.info("Replanning based on feedback")
            revised_plan = self.replan_chain.invoke({
                "original_plan": original_plan.json(),
                "feedback": feedback
            })
            self.logger.info("Plan revised successfully")
            return revised_plan
        except Exception as e:
            self.logger.error(f"Error replanning: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    classifier = IntentClassifier("deepseek-chat", openai_api_base='https://api.deepseek.com',
                                  openai_api_key=os.environ["DEEPSEEK_API_KEY"])
    planner = TaskPlanner(classifier, "deepseek-chat", openai_api_base='https://api.deepseek.com',
                          openai_api_key=os.environ["DEEPSEEK_API_KEY"])

    task = "Analyze our Q2 sales performance and visualize the top-performing products."
    intent = "data_analysis"
    capabilities = ["data_query", "data_analysis", "data_visualization"]

    initial_plan = planner.create_plan(task, intent, capabilities, "")
    if initial_plan:
        print("Initial Plan:")
        print(json.dumps(json.loads(initial_plan.json()), indent=2))

        feedback = "We need to include a comparison with Q1 performance in the analysis."
        revised_plan = planner.replan(initial_plan, feedback)
        if revised_plan:
            print("\nRevised Plan:")
            print(json.dumps(json.loads(revised_plan.json()), indent=2))
    else:
        print("Failed to create initial plan.")
