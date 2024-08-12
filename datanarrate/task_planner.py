import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from context_manager import ContextManager
from query_analyzer import QueryAnalysis


class DataSource(BaseModel):
    name: str = Field(description="Name of the data source (e.g., 'mysql' or 'elasticsearch')")
    tables_or_indices: List[str] = Field(description="List of relevant tables or indices for this data source")
    fields: Dict[str, List[str]] = Field(description="Suggested fields for each table or index")


class TaskStep(BaseModel):
    step_number: int = Field(description="The order of the step in the plan")
    description: str = Field(description="A clear, concise description of the step")
    required_capability: str = Field(description="The high-level capability required for this step")
    input_description: Dict[str, Any] = Field(description="Description of required inputs for this step", default={})
    tools: List[str] = Field(description="List of tools that might be useful for this step", default=[])
    data_sources: List[DataSource] = Field(description="List of relevant data sources for this step", default=[])


class TaskPlan(BaseModel):
    steps: List[TaskStep] = Field(description="The list of steps in the task plan")
    reasoning: str = Field(description="Explanation of the planning process")


class TaskPlanner:
    def __init__(self, llm: BaseChatModel, context_manager: ContextManager, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=TaskPlan)
        self.plan_chain = self._create_plan_chain()

    def _create_plan_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a task planner for a data analysis system. "
                       "Given a query analysis and context, create a detailed plan to accomplish the task. "
                       "Consider the task type, sub-tasks, relevant intents, and potential tools. "
                       "Use the provided relevant tables/indices and suggested fields to create more specific and efficient steps. "
                       "For each step, specify which data sources, tables/indices, and fields are relevant. "
                       "The context includes a list of current intents and their confidences, as well as schema information. "
                       "Ensure each step is clear, actionable, and makes use of the specific data sources identified.\n\n"
                       "Unified Schema Compression Format Explanation:\n"
                       "- MySQL tables: 'mysql': {{'table_name': ['column:typ?*', ...]}}\n"
                       "  where 'typ' is the first 3 characters of the data type,\n"
                       "  '?' indicates a nullable column, and '*' indicates a primary key.\n"
                       "- Elasticsearch indices: 'elasticsearch': {{'index_name': {{'field': 'typ', ...}}}}\n"
                       "  where 'typ' is the first 3 characters of the field type.\n"
                       "When referring to tables/indices and fields in your plan, use the actual names from this compressed schema.\n"
                       "Output format: {format_instructions}"),
            ("human", "Query Analysis: {query_analysis}\nContext: {context}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        return prompt | self.llm | self.output_parser

    def plan_task(self, query_analysis: QueryAnalysis, compressed_schema: Dict[str, Any]) -> Optional[TaskPlan]:
        try:
            self.logger.info("Planning task based on query analysis")
            context = self.context_manager.get_context_summary()
            context["schema_info"] = compressed_schema
            context["relevant_tables"] = query_analysis.required_data_sources
            context["suggested_fields"] = {ds.name: ds.suggested_fields for ds in query_analysis.required_data_sources}

            plan = self.plan_chain.invoke({
                "query_analysis": json.dumps(query_analysis.dict(), default=str),
                "context": json.dumps(context, default=str)
            })
            self.logger.info(f"Task plan created: {plan}")
            return plan
        except Exception as e:
            self.logger.error(f"Error planning task: {e}", exc_info=True)
            return None

    def replan(self, previous_plan: TaskPlan, feedback: str) -> Optional[TaskPlan]:
        try:
            self.logger.info("Replanning task based on feedback")
            context = self.context_manager.get_context_summary()
            replan_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a task planner for a data analysis system. "
                           "Given a previous plan, feedback, and updated context, create an updated plan. "
                           "Consider the feedback carefully and adjust the plan accordingly. "
                           "Output format: {format_instructions}"),
                ("human", "Previous Plan: {previous_plan}\nFeedback: {feedback}\nContext: {context}")
            ]).partial(format_instructions=self.output_parser.get_format_instructions())

            replan_chain = replan_prompt | self.llm | self.output_parser
            updated_plan = replan_chain.invoke({
                "previous_plan": json.dumps(previous_plan.dict(), default=str),
                "feedback": feedback,
                "context": json.dumps(context, default=str)
            })
            self.logger.info(f"Updated task plan created: {updated_plan}")
            return updated_plan
        except Exception as e:
            self.logger.error(f"Error replanning task: {e}", exc_info=True)
            return None

    def update_context_with_plan(self, plan: TaskPlan):
        """
        Update the context manager with the current plan.
        """
        self.context_manager.update_state(current_task=plan.steps[0].description if plan.steps else "")
        self.context_manager.add_to_conversation_history("assistant",
                                                         f"I've created a plan with {len(plan.steps)} steps.")
        self.logger.info("Updated context with new plan")


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from query_analyzer import QueryAnalyzer
    from intent_classifier import IntentClassifier
    from config import config

    # Set up logging
    logging.basicConfig(level=config.LOG_LEVEL)

    # Initialize LLM
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.2
    )

    # Initialize IntentClassifier, ContextManager, QueryAnalyzer, and TaskPlanner
    intent_classifier = IntentClassifier(llm)
    context_manager = ContextManager(intent_classifier, thread_id="example_thread")
    query_analyzer = QueryAnalyzer(llm)
    task_planner = TaskPlanner(llm, context_manager)

    # Test the TaskPlanner
    test_query = "Show me a bar chart of our top 5 selling products in Q2, including their revenue and compare it with last year's Q2 performance"

    # Update context with the test query
    context_manager.update_context(test_query)

    # Analyze the query
    query_analysis = query_analyzer.analyze_query(test_query, context_manager.get_state().current_intents,
                                                  context_manager.get_state().relevant_data.get("schema_info", {}))

    if query_analysis:
        task_plan = task_planner.plan_task(query_analysis,
                                           context_manager.get_state().relevant_data.get("schema_info", {}))
        if task_plan:
            print("Initial Task Plan:")
            for step in task_plan.steps:
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Required Capability: {step.required_capability}")
                print(f"  Tools: {', '.join(step.tools)}")
                print(f"  Data Sources:")
                for data_source in step.data_sources:
                    print(f"    - {data_source.name}:")
                    print(f"        Tables/Indices: {', '.join(data_source.tables_or_indices)}")
                    print(f"        Fields:")
                    for table_or_index, fields in data_source.fields.items():
                        print(f"          - {table_or_index}: {', '.join(fields)}")
            print(f"\nReasoning: {task_plan.reasoning}")

            # Update context with the initial plan
            task_planner.update_context_with_plan(task_plan)

            # Test replanning
            feedback = "The plan looks good, but we need to add a step to check for any data anomalies before visualization."
            context_manager.add_to_conversation_history("user", feedback)

            updated_plan = task_planner.replan(task_plan, feedback)
            if updated_plan:
                print("\nUpdated Task Plan:")
                for step in updated_plan.steps:
                    print(f"Step {step.step_number}: {step.description}")
                    print(f"  Required Capability: {step.required_capability}")
                    print(f"  Tools: {', '.join(step.tools)}")
                    print(f"  Data Sources:")
                    for data_source in step.data_sources:
                        print(f"    - {data_source.name}:")
                        print(f"        Tables/Indices: {', '.join(data_source.tables_or_indices)}")
                        print(f"        Fields:")
                        for table_or_index, fields in data_source.fields.items():
                            print(f"          - {table_or_index}: {', '.join(fields)}")
                print(f"\nReasoning: {updated_plan.reasoning}")

                # Update context with the updated plan
                task_planner.update_context_with_plan(updated_plan)

            # Print final context summary
            print("\nFinal Context Summary:")
            print(json.dumps(context_manager.get_context_summary(), indent=2, default=str))
        else:
            print("Task planning failed.")
    else:
        print("Query analysis failed.")
