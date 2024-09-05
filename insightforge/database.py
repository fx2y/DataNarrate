from typing import Dict, Any, Tuple

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from datanarrate.config import config
from insightforge.state import State


def create_sql_database_toolkit() -> SQLDatabaseToolkit:
    """
    Establish a connection to a MySQL database and create an SQLDatabaseToolkit instance.
    """
    # Establish connection to the MySQL database
    db = SQLDatabase.from_uri(
        config.SQLALCHEMY_DATABASE_URI,
        sample_rows_in_table_info=3
    )
    llm = ChatOpenAI(model=config.LLM_MODEL_NAME, openai_api_base=config.OPENAI_API_BASE,
                     openai_api_key=config.OPENAI_API_KEY, temperature=0)

    # Create and return the SQLDatabaseToolkit instance
    return SQLDatabaseToolkit(db=db, llm=llm)


def retrieve_schema(toolkit: SQLDatabaseToolkit) -> Dict[str, Any]:
    list_tables_tool = next(tool for tool in toolkit.get_tools() if tool.name == "sql_db_list_tables")
    get_schema_tool = next(tool for tool in toolkit.get_tools() if tool.name == "sql_db_schema")

    tables = list_tables_tool.invoke("")
    schema = {}

    for table in tables.split(", "):
        table_schema = get_schema_tool.invoke(table)
        schema[table] = table_schema

    return {"schema": schema}


def update_state_with_schema(state: State) -> State:
    toolkit = create_sql_database_toolkit()
    schema_info = retrieve_schema(toolkit)
    state["schema"] = schema_info["schema"]
    return state


class DecisionOutput(BaseModel):
    """Output schema for the decision function."""
    decision: str = Field(description="Either 'clarify' or 'query'")
    clarification_question: str = Field(description="The clarification question if decision is 'clarify'", default="")
    sql_query: str = Field(description="The SQL query if decision is 'query'", default="")


def decide_action(state: State) -> Tuple[str, str]:
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI assistant that decides whether to ask for clarification or craft a SQL query based on a user's request and database schema. Respond with either a clarification question or a SQL query."),
        ("human", "Database schema: {schema}"),
        ("human", "User request: {request}"),
        ("human",
         "Decide whether to ask for clarification or craft a SQL query. Use <clarification></clarification> tags for a clarification question or <sql></sql> tags for a SQL query.")
    ])

    response = llm.invoke(prompt.format(
        schema=state["schema"],
        request=state["messages"][-1].content
    ))

    if "<clarification>" in response.content:
        decision = "clarify"
        output = response.content.split("<clarification>")[1].split("</clarification>")[0].strip()
    elif "<sql>" in response.content:
        decision = "query"
        output = response.content.split("<sql>")[1].split("</sql>")[0].strip()
    else:
        raise ValueError("LLM response does not contain expected tags")

    return decision, output


def update_state_with_decision(state: State) -> State:
    decision, output = decide_action(state)
    if decision == "clarify":
        state["messages"] = [AIMessage(output)]
    else:
        state["query"] = output
    return state


def test_decide_action():
    # Set up the initial state
    toolkit = create_sql_database_toolkit()
    schema_info = retrieve_schema(toolkit)

    state = State(
        messages=[],
        schema=schema_info["schema"],
        query=None,
        results=None,
        visualization_data=None,
        narration=None
    )

    # Test cases
    test_cases = [
        "Create a line chart showing the economic growth trends for all provinces from 2018 to 2023.",
        "What's the weather like today?",
        "List the top 5 customers by revenue",
    ]

    for case in test_cases:
        print(f"\nTest case: {case}")
        state["messages"] = [HumanMessage(case)]
        decision, output = decide_action(state)
        print(f"Decision: {decision}")
        if decision == 'clarify':
            print(f"Clarification question: {output}")
        else:
            print(f"SQL query: {output}")


if __name__ == "__main__":
    test_decide_action()
