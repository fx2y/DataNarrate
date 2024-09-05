from typing import Annotated, TypedDict, Literal
from typing import Optional

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from datanarrate.config import config as cfg


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_request: str
    schema: str
    query: str
    results: list
    visualization_data: dict
    narration: str
    need_human_input: bool


class DatabaseConfig(BaseModel):
    uri: str = Field(..., description="Database connection URI")
    include_tables: Optional[list[str]] = Field(None, description="List of tables to include")
    sample_rows_in_table_info: int = Field(2, description="Number of sample rows to include in table info")


def create_sql_database_tool(config: DatabaseConfig, llm: BaseLanguageModel) -> SQLDatabaseToolkit:
    """
    Establishes a connection to a MySQL database and creates an SQLDatabaseToolkit instance.

    Args:
        config (DatabaseConfig): Configuration for the database connection.

    Returns:
        SQLDatabaseTool: An instance of SQLDatabaseTool for interacting with the database.

    Raises:
        ValueError: If there's an error connecting to the database or creating the tool.
    """
    try:
        db = SQLDatabase.from_uri(
            config.uri,
            include_tables=config.include_tables,
            sample_rows_in_table_info=config.sample_rows_in_table_info
        )
        return SQLDatabaseToolkit(db=db, llm=llm)
    except Exception as e:
        raise ValueError(f"Error creating SQLDatabaseTool: {str(e)}") from e


def get_database_schema(toolkit: SQLDatabaseToolkit, config: RunnableConfig) -> dict:
    """
    Retrieves the database schema using the SQLDatabaseToolkit and returns it as part of the State.

    Args:
        toolkit (SQLDatabaseToolkit): The SQLDatabaseToolkit instance.
        config (RunnableConfig): Configuration for the runnable.

    Returns:
        dict: A dictionary containing the updated State with the schema.

    Raises:
        ValueError: If there's an error retrieving the schema.
    """
    try:
        list_tables_tool = next(tool for tool in toolkit.get_tools() if tool.name == "sql_db_list_tables")
        get_schema_tool = next(tool for tool in toolkit.get_tools() if tool.name == "sql_db_schema")

        tables = list_tables_tool.invoke("")
        schema = ""
        for table in tables.split(", "):
            schema += f"{table}:\n{get_schema_tool.invoke(table)}\n\n"

        return {"schema": schema}
    except Exception as e:
        raise ValueError(f"Error retrieving database schema: {str(e)}") from e


class DecisionOutput(TypedDict):
    decision: Literal["clarify", "query"]
    content: str


def analyze_request(user_request: str, db_schema: str, model: ChatOpenAI) -> DecisionOutput:
    """
    Analyzes the user's request and database schema to decide whether to ask for clarification or craft a SQL query.

    Args:
        user_request (str): The user's natural language request.
        db_schema (str): The database schema.
        model (ChatOpenAI): The language model to use for decision-making.

    Returns:
        DecisionOutput: A dictionary containing the decision and either a clarification question or SQL query.

    Raises:
        ValueError: If there's an error in processing the request.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI assistant that analyzes user requests and database schemas to decide whether to ask for clarification or craft a SQL query. If the request is clear and can be answered with the given schema, craft a SQL query. Otherwise, ask for clarification."),
        ("human",
         "User request: {request}\n\nDatabase schema:\n{schema}\n\nDecide whether to clarify or query, and provide the appropriate response."),
    ])

    try:
        response = model.invoke(prompt.format(request=user_request, schema=db_schema))

        if "CLARIFY:" in response.content:
            return {"decision": "clarify", "content": response.content.split("CLARIFY:", 1)[1].strip()}
        elif "SQL:" in response.content:
            return {"decision": "query", "content": response.content.split("SQL:", 1)[1].strip()}
        else:
            raise ValueError("Unexpected response format from the model")

    except Exception as e:
        raise ValueError(f"Error processing request: {str(e)}") from e


def analyze_request_node(state: State) -> dict:
    decision = analyze_request(state["user_request"], state["schema"], model)
    if decision["decision"] == "clarify":
        return {"messages": [AIMessage(content=decision["content"])]}
    else:
        return {"query": decision["content"]}


def clarify_request_node(state: State) -> dict:
    clarification_response = state["messages"][-1].content
    new_request = f"{state['user_request']} (Clarification: {clarification_response})"
    return {
        "user_request": new_request,
        "need_human_input": True
    }


def human_input_node(state: State) -> dict:
    print("Clarification needed. Current request:", state["user_request"])
    user_input = input("Please provide additional information: ")
    return {
        "user_request": f"{state['user_request']} (Human input: {user_input})",
        "need_human_input": False,
        "messages": [HumanMessage(content=user_input)]
    }


def should_clarify(state: State) -> Literal["clarify", "execute_query"]:
    if state.get("query"):
        return "execute_query"
    return "clarify"


def should_get_human_input(state: State) -> Literal["human_input", "analyze"]:
    if state["need_human_input"]:
        return "human_input"
    return "analyze"


def create_graph(toolkit: SQLDatabaseToolkit, llm: BaseLanguageModel) -> CompiledStateGraph:
    workflow = StateGraph(State)

    # Add the get_database_schema node
    workflow.add_node("get_schema", lambda state, config: get_database_schema(toolkit, config))

    workflow.add_node("analyze", analyze_request_node)
    workflow.add_node("clarify", clarify_request_node)
    workflow.add_node("human_input", human_input_node)

    # Add other nodes (placeholder functions for now)
    workflow.add_node("generate_query", lambda state: {"query": "SELECT * FROM example_table"})
    workflow.add_node("execute_query", lambda state: {"results": [{"column1": "value1"}]})
    workflow.add_node("visualize_results", lambda state: {"visualization_data": {"type": "bar_chart"}})
    workflow.add_node("generate_narration", lambda state: {"narration": "Here's what the data shows..."})

    # Define the edges
    workflow.add_edge("get_schema", "analyze")
    workflow.add_conditional_edges("analyze", should_clarify)
    workflow.add_conditional_edges("clarify", should_get_human_input)
    workflow.add_edge("generate_query", "execute_query")
    workflow.add_edge("execute_query", "visualize_results")
    workflow.add_edge("visualize_results", "generate_narration")
    workflow.add_edge("generate_narration", END)

    # Set the entry point
    workflow.set_entry_point("get_schema")

    return workflow.compile(checkpointer=MemorySaver(), interrupt_before=["action"])


if __name__ == '__main__':
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase
    from langchain_core.runnables import RunnableConfig


    def test_get_database_schema_integration():
        llm = ChatOpenAI(model=cfg.LLM_MODEL_NAME, openai_api_base=cfg.OPENAI_API_BASE,
                         openai_api_key=cfg.OPENAI_API_KEY, temperature=0)

        # Create a real SQLite database
        db = SQLDatabase.from_uri("sqlite:///test.db")

        # Create some test tables
        db.run("CREATE TABLE IF NOT EXISTS test_table1 (id INTEGER PRIMARY KEY, name TEXT)")
        db.run("CREATE TABLE IF NOT EXISTS test_table2 (id INTEGER PRIMARY KEY, value REAL)")

        # Create the toolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # Call the function
        result = get_database_schema(toolkit, RunnableConfig())

        # Assert the result
        assert "test_table1" in result["schema"]
        assert "test_table2" in result["schema"]
        assert "id INTEGER" in result["schema"]
        assert "name TEXT" in result["schema"]
        assert "value REAL" in result["schema"]

        # Clean up
        db.run("DROP TABLE test_table1")
        db.run("DROP TABLE test_table2")


    test_get_database_schema_integration()

    llm = ChatOpenAI(model=cfg.LLM_MODEL_NAME, openai_api_base=cfg.OPENAI_API_BASE,
                     openai_api_key=cfg.OPENAI_API_KEY, temperature=0)
    config = DatabaseConfig(uri=cfg.SQLALCHEMY_DATABASE_URI)
    toolkit = create_sql_database_tool(config, llm)
    graph = create_graph(toolkit, llm)
    config = {"configurable": {"thread_id": "thread-1"}}

    result = graph.invoke({"messages": [HumanMessage(
        content="Create a line chart showing the economic growth trends for all provinces from 2018 to 2023.")]},
        config=config)
    snapshot = graph.get_state(config)
