from typing import Annotated, Literal, List, Optional

from langchain_core.messages import AnyMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.constants import END, START
from langgraph.graph import add_messages, StateGraph
from typing_extensions import TypedDict

from sql_agent.tool import get_sql_tools, get_db_query_tool
from sql_agent.util import create_tool_node_with_fallback


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    schema: Optional[str]
    query: Optional[str]
    results: Optional[str]
    visualization_data: Optional[dict]
    narration: Optional[str]


def create_query_gen(llm):
    query_gen_system = """You are a SQL expert with a strong attention to detail.

    Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

    DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

    When generating the query:

    Output the SQL query that answers the input question without a tool call.

    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.

    If you get an error while executing a query, rewrite the query and try again.

    If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
    NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

    If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
    query_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", query_gen_system), ("placeholder", "{messages}")]
    )
    return query_gen_prompt | llm.bind_tools(
        [SubmitFinalAnswer]
    )


# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "sql_db_list_tables",
                    "args": {},
                    "id": "tool_abcd123",
                }],
            )
        ]
    }


# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")


# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"


def create_workflow(llm, db):
    list_tables_tool, get_schema_tool = get_sql_tools(db, llm)

    db_query_tool = get_db_query_tool(db)

    query_gen = create_query_gen(llm)

    def create_query_check(llm):
        query_check_system = """You are a SQL expert with a strong attention to detail. Double check the SQLite query for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins
        If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.
        You will call the appropriate tool to execute the query after running this check."""

        query_check_prompt = ChatPromptTemplate.from_messages([
            ("system", query_check_system),
            ("placeholder", "{messages}")
        ])

        return query_check_prompt | llm.bind_tools(
            [db_query_tool],
            tool_choice="required"
        )

    def query_gen_node(state: State):
        message = query_gen.invoke(state)

        # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"],
                        )
                    )
        else:
            tool_messages = []
        return {"messages": [message] + tool_messages}

    query_check = create_query_check(llm)

    def model_check_query(state: State) -> dict[str, list[AIMessage]]:
        """
        Use this tool to double-check if your query is correct before executing it.
        """
        return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}

    model_get_schema = llm.bind_tools(
        [get_schema_tool]
    )

    workflow = StateGraph(State)

    workflow.add_node("first_tool_call", first_tool_call)
    workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
    workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
    workflow.add_node(
        "model_get_schema",
        lambda state: {
            "messages": [model_get_schema.invoke(state["messages"])],
        },
    )
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("correct_query", model_check_query)
    workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

    # Specify the edges between the nodes
    workflow.add_edge(START, "first_tool_call")
    workflow.add_edge("first_tool_call", "list_tables_tool")
    workflow.add_edge("list_tables_tool", "model_get_schema")
    workflow.add_edge("model_get_schema", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "query_gen")
    workflow.add_conditional_edges(
        "query_gen",
        should_continue,
    )
    workflow.add_edge("correct_query", "execute_query")
    workflow.add_edge("execute_query", "query_gen")

    return workflow.compile()
