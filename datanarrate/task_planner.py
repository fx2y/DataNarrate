import json
import logging
from typing import List, Dict, Any, Optional, Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import FunctionMessage, BaseMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBranch, chain as as_runnable
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START

from query_analyzer import QueryAnalysis


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
    selected_tool: str = Field(description="The name of the selected tool for this step")
    tool_input: Dict[str, Any] = Field(description="Input parameters for the selected tool")
    data_sources: List[DataSource] = Field(description="List of relevant data sources for this step",
                                           default_factory=list)
    query_info: Optional[QueryInfo] = Field(description="Detailed information about the required query", default=None)


class TaskPlan(BaseModel):
    steps: List[TaskStep] = Field(description="The list of steps in the task plan")
    reasoning: str = Field(description="Explanation of the planning process")


class TaskPlannerState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    query_analysis: Optional[QueryAnalysis] = None
    task_plan: Optional[TaskPlan] = None
    current_step: int = 0
    status: str = "in_progress"


class TaskPlanner:
    def __init__(self, llm: BaseChatModel, tools: List[BaseTool], logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.output_parser = PydanticOutputParser(pydantic_object=TaskPlan)
        self.planner = self._create_planner()
        self.tools = tools
        self.graph = self._create_graph()

    def _create_planner(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a task planner for a data analysis system. "
                       "Given a query analysis and context, create a detailed plan to accomplish the task. "
                       "Consider the task type, sub-tasks, relevant intents, and potential tools. "
                       "Use the provided relevant tables/indices and suggested fields to create more specific and efficient steps. "
                       "For each step, specify which data sources, tables/indices, and fields are relevant. "
                       "For database-related steps, include detailed query information in the query_info field. "
                       "For each step, select the most appropriate tool and provide the necessary input based on the tool's schema. "
                       "Available tools: {tools}\n\n"
                       "Output format: {format_instructions}"),
            ("human", "Query Analysis: {query_analysis}\nContext: {context}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())

        def should_replan(state: list):
            return isinstance(state[-1], SystemMessage)

        def wrap_messages(state: list):
            return {"messages": state}

        return (
                RunnableBranch(
                    (should_replan, wrap_messages | prompt),
                    wrap_messages | prompt,
                )
                | self.llm.bind_tools(self.tools)
                | self.output_parser
        )

    def _create_graph(self):
        graph = StateGraph(TaskPlannerState)

        graph.add_node("plan", self._plan_task)
        graph.add_node("execute_step", self._execute_step)
        graph.add_node("evaluate", self._evaluate_step)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute_step")
        graph.add_edge("execute_step", "evaluate")

        graph.add_conditional_edges(
            "evaluate",
            self._route_next_step,
            {
                "execute_step": "execute_step",
                "plan": "plan",
                END: END
            }
        )

        return graph.compile()

    def _plan_task(self, state: TaskPlannerState) -> Dict[str, Any]:
        try:
            self.logger.info("Planning task based on query analysis")
            context = {
                "schema_info": state.query_analysis.schema_info,
                "relevant_tables": state.query_analysis.required_data_sources,
                "suggested_fields": {ds.name: ds.suggested_fields for ds in state.query_analysis.required_data_sources}
            }

            plan = self.planner.invoke({
                "query_analysis": state.query_analysis.dict(),
                "context": context,
                "tools": [tool.description for tool in self.tools],
                "messages": state.messages
            })

            self.logger.info(f"Task plan created: {plan}")
            return {"task_plan": plan}
        except Exception as e:
            self.logger.error(f"Error planning task: {e}", exc_info=True)
            return {}

    def _execute_step(self, state: TaskPlannerState) -> Dict[str, Any]:
        current_step = state.task_plan.steps[state.current_step]
        tool = next((t for t in self.tools if t.name == current_step.selected_tool), None)
        if tool:
            try:
                result = tool.invoke(current_step.tool_input)
                return {
                    "messages": state.messages + [FunctionMessage(content=str(result), name=tool.name)],
                    "current_step": state.current_step + 1
                }
            except Exception as e:
                self.logger.error(f"Error executing step: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}
        else:
            self.logger.error(f"Tool not found: {current_step.selected_tool}")
            return {"status": "error", "error": f"Tool not found: {current_step.selected_tool}"}

    def _evaluate_step(self, state: TaskPlannerState) -> Dict[str, Any]:
        if state.current_step >= len(state.task_plan.steps):
            return {"status": "complete"}
        elif state.status == "error":
            return {"status": "replan"}
        else:
            return {"status": "continue"}

    def _route_next_step(self, state: TaskPlannerState) -> str:
        if state.status == "complete":
            return END
        elif state.status == "replan":
            return "plan"
        else:
            return "execute_step"

    @as_runnable
    def run(self, query_analysis: QueryAnalysis) -> Iterable[Dict[str, Any]]:
        state = TaskPlannerState(query_analysis=query_analysis)
        while True:
            state = self.graph.invoke(state)
            yield state
            if state.status == "complete":
                break

    def update_context_with_plan(self, plan: TaskPlan):
        self.context_manager.add_to_conversation_history("system", f"Task plan created: {plan.dict()}")


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
    query_analyzer = QueryAnalyzer(llm)


    # Register tools (example)
    class SQLQueryTool(BaseTool):
        name = "SQL Query Tool"
        description = "Executes SQL queries on a MySQL database"

        def _run(self, query: str) -> str:
            return f"Executed SQL query: {query}"


    task_planner = TaskPlanner(llm)

    # Test the TaskPlanner
    test_query = "Show me a bar chart of our top 5 selling products in Q2, including their revenue and compare it with last year's Q2 performance"

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

    # Analyze the query
    intents = intent_classifier.classify(test_query)
    query_analysis = query_analyzer.analyze_query(test_query, intents.intents, compressed_schema)

    if query_analysis:
        task_plan = task_planner.run(query_analysis)
        if task_plan:
            print("Task Plan:")
            for step in task_plan.steps:
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Required Capability: {step.required_capability}")
                print(f"  Selected Tool: {step.selected_tool}")
                print(f"  Tool Input: {json.dumps(step.tool_input, indent=2)}")
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

            updated_plan = task_planner.run(query_analysis)
            if updated_plan:
                print("\nUpdated Task Plan:")
                for step in updated_plan.steps:
                    print(f"Step {step.step_number}: {step.description}")
                    print(f"  Required Capability: {step.required_capability}")
                    print(f"  Selected Tool: {step.selected_tool}")
                    print(f"  Tool Input: {json.dumps(step.tool_input, indent=2)}")
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
                print(f"\nReasoning: {updated_plan.reasoning}")

                # Update context with the updated plan
                task_planner.update_context_with_plan(updated_plan)

            # Print final context summary
            print("\nFinal Context Summary:")
        else:
            print("Task planning failed.")
    else:
        print("Query analysis failed.")
