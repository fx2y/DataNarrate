import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator

from context_manager import ContextManager
from query_analyzer import QueryAnalysis
from query_validator import QueryValidator


class DataSource(BaseModel):
    name: str = Field(description="Name of the data source (e.g., 'mysql' or 'elasticsearch')")
    tables_or_indices: List[str] = Field(description="List of relevant tables or indices for this data source")
    fields: Dict[str, List[str]] = Field(description="Suggested fields for each table or index")


class QueryInfo(BaseModel):
    data_source: str = Field(description="The data source for this query (e.g., 'mysql' or 'elasticsearch')")
    query_type: str = Field(description="The type of query (e.g., 'SELECT', 'AGGREGATE', 'JOIN', 'NESTED')")
    tables_or_indices: List[str] = Field(description="The tables or indices involved in this query")
    fields: List[str] = Field(description="The fields to be queried or returned")
    filters: Optional[Dict[str, Any]] = Field(description="Any filters or conditions to be applied", default=None)
    aggregations: Optional[List[str]] = Field(description="Any aggregations to be performed", default=None)
    order_by: Optional[List[str]] = Field(description="Fields to order the results by", default=None)
    limit: Optional[int] = Field(description="Limit on the number of results", default=None)


class TaskStep(BaseModel):
    step_number: int = Field(description="The order of the step in the plan")
    description: str = Field(description="A clear, concise description of the step")
    required_capability: str = Field(description="The high-level capability required for this step")
    input_description: Dict[str, Any] = Field(description="Description of required inputs for this step",
                                              default_factory=dict)
    tools: List[str] = Field(description="List of tools that might be useful for this step", default_factory=list)
    data_sources: List[DataSource] = Field(description="List of relevant data sources for this step",
                                           default_factory=list)
    query_info: Optional[QueryInfo] = Field(description="Detailed information about the required query", default=None)

    @validator('query_info', pre=True)
    def handle_empty_query_info(cls, v):
        if v == {}:
            return None
        return v

    @validator('input_description', pre=True)
    def handle_input_description(cls, v):
        if v is None or v == "None":
            return {}
        if isinstance(v, str):
            return {"description": v}
        return v

    @validator('data_sources', pre=True)
    def handle_data_sources(cls, v):
        if v is None:
            return []
        return v

    class Config:
        extra = 'allow'


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
        self.query_validator = QueryValidator(logger=logger)

    def _create_plan_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a task planner for a data analysis system. "
                       "Given a query analysis and context, create a detailed plan to accomplish the task. "
                       "Consider the task type, sub-tasks, relevant intents, and potential tools. "
                       "Use the provided relevant tables/indices and suggested fields to create more specific and efficient steps. "
                       "For each step, specify which data sources, tables/indices, and fields are relevant. "
                       "For database-related steps, include detailed query information in the query_info field. "
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
                "query_analysis": query_analysis.dict(),
                "context": context
            })

            # Post-process the plan to ensure task descriptions and query info are included for database steps
            self._post_process_plan(plan, query_analysis, compressed_schema)

            self.logger.info(f"Task plan created: {plan}")
            return plan
        except Exception as e:
            self.logger.error(f"Error planning task: {e}", exc_info=True)
            return None

    def _post_process_plan(self, plan: TaskPlan, query_analysis: QueryAnalysis, compressed_schema: Dict[str, Any]):
        for step in plan.steps:
            if isinstance(step.input_description, str):
                step.input_description = {"description": step.input_description}
            elif step.input_description is None:
                step.input_description = {}

            if step.data_sources is None:
                step.data_sources = []

            if any(tool.lower().startswith(("sql", "elasticsearch")) for tool in step.tools):
                step.input_description["task"] = step.description
                step.input_description["query_context"] = query_analysis.task_type
                if query_analysis.time_range:
                    step.input_description["time_range"] = query_analysis.time_range

    def replan(self, original_plan: TaskPlan, feedback: str, current_step: int) -> TaskPlan:
        """
        Revise the plan based on feedback and new context.
        """
        try:
            self.logger.info(f"Replanning based on feedback: {feedback}")
            prompt = ChatPromptTemplate.from_messages([("system", """"
            You are an AI task planner. You have been given an original plan and feedback about its execution.
            Your task is to revise the plan, taking into account the feedback and the current execution state.

            Original Plan:
            {original_plan}

            Feedback:
            {feedback}

            Current Step: {current_step}

            Please create a revised plan. You can:
            1. Modify existing steps
            2. Add new steps
            3. Remove steps
            4. Reorder steps

            Focus on addressing the issues raised in the feedback, but also consider the overall efficiency and effectiveness of the plan.
            If possible, try to salvage and reuse parts of the original plan that are still valid.

            Provide the revised plan in the same format as the original plan, along with a brief explanation of your changes.
            """)]).partial(format_instructions=self.output_parser.get_format_instructions())

            replan_chain = prompt | self.llm | self.output_parser
            new_plan = replan_chain.invoke({
                "original_plan": json.dumps(original_plan.dict(), indent=2),
                "feedback": feedback,
                "current_step": current_step
            })

            self.logger.info(f"Generated revised plan with {len(new_plan.steps)} steps")
            self.logger.info(f"Replanning reasoning: {new_plan.reasoning}")

            return new_plan
        except Exception as e:
            self.logger.error(f"Error in replan: {e}", exc_info=True)
            return original_plan  # Return the original plan if replanning fails

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
    query_analyzer = QueryAnalyzer(llm, context_manager)
    task_planner = TaskPlanner(llm, context_manager)

    # Test the TaskPlanner
    test_query = "Show me a bar chart of our top 5 selling products in Q2, including their revenue and compare it with last year's Q2 performance"

    # Update context with the test query
    context_manager.update_context(test_query)

    compressed_schema = {
        "mysql": {
            "sales": ["date:dat", "product_id:int*", "customer_id:int", "quantity:int", "revenue:dec"],
            "products": ["product_id:int*", "name:var", "category:var", "price:dec"],
            "customers": ["customer_id:int*", "name:var", "location:var"]
        },
        "elasticsearch": {
            "orders_index": {
                "order_id": "key",
                "date": "dat",
                "customer": "nes",
                "customer.id": "key",
                "customer.name": "tex",
                "customer.email": "tex",
                "items": "nes",
                "items.product_id": "key",
                "items.name": "tex",
                "items.quantity": "int",
                "items.price": "flo",
                "total_amount": "flo"
            },
            "product_index": {
                "product_id": "key",
                "name": "tex",
                "category": "key",
                "price": "flo",
                "specifications": "nes",
                "specifications.brand": "tex",
                "specifications.model": "tex",
                "specifications.year": "int"
            }
        }
    }

    context_manager.update_schema_info(compressed_schema)

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
                if step.query_info:
                    print(f"  Query Info:")
                    print(f"    - Data Source: {step.query_info.data_source}")
                    print(f"    - Query Type: {step.query_info.query_type}")
                    print(f"    - Tables/Indices: {', '.join(step.query_info.tables_or_indices)}")
                    print(f"    - Fields: {', '.join(step.query_info.fields)}")
                    if step.query_info.filters:
                        print(f"    - Filters: {step.query_info.filters}")
                    if step.query_info.aggregations:
                        print(f"    - Aggregations: {', '.join(step.query_info.aggregations)}")
                    if step.query_info.order_by:
                        print(f"    - Order By: {', '.join(step.query_info.order_by)}")
                    if step.query_info.limit:
                        print(f"    - Limit: {step.query_info.limit}")
            print(f"\nReasoning: {task_plan.reasoning}")

            # Update context with the initial plan
            task_planner.update_context_with_plan(task_plan)

            # Test replanning
            feedback = "The plan looks good, but we need to add a step to check for any data anomalies before visualization."
            context_manager.add_to_conversation_history("user", feedback)

            updated_plan = task_planner.replan(task_plan, feedback, current_step=1)
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
