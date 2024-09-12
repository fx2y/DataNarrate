import ast
import json
import logging
import re
import uuid
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Union

import pandas as pd
import plotly.graph_objs as go
from fastapi import FastAPI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langserve import add_routes

from datanarrate.config import config
from insightforge.state import State

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """
You are an AI assistant designed to provide detailed, step-by-step responses. Your outputs should follow this structure:

1. Begin with a <thinking> section. Everything in this section is invisible to the user.
2. Inside the thinking section:
 a. Briefly analyze the question and outline your approach.
 b. Present a clear plan of steps to solve the problem.
 c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.
3. Include a <reflection> section for each idea where you:
 a. Review your reasoning.
 b. Check for potential errors or oversights.
 c. Confirm or adjust your conclusion if necessary.
4. Be sure to close all reflection sections.
5. Close the thinking section with </thinking>.
6. Provide your final answer in an <output> section.

Always use these tags in your responses. Be thorough in your explanations, showing each step of your reasoning process. Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. Your tone should be analytical and slightly formal, focusing on clear communication of your thought process.

Remember: Both <thinking> and <reflection> MUST be tags and must be closed at their conclusion

Make sure all <tags> are on separate lines with no other text. Do not include other text on a line containing a tag.
"""


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


def update_state_with_schema(state: State, toolkit: SQLDatabaseToolkit) -> State:
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
        ("system", REFLECTION_SYSTEM_PROMPT),
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
        state["messages"] += [AIMessage(output)]
    else:
        state["query"] = output
    return state


def execute_query(state: State, toolkit: SQLDatabaseToolkit) -> State:
    """
    Execute the SQL query using the SQLDatabaseToolkit and return the results as part of the State.
    """
    if not state.get("query"):
        raise ValueError("No SQL query found in the state")

    # Get the SQL database query tool
    db_query_tool = next(tool for tool in toolkit.get_tools() if tool.name == "sql_db_query")

    try:
        # Execute the query
        results = db_query_tool.invoke(state["query"])

        # Update the state with the query results
        state["results"] = results

        # Log the successful query execution
        logger.info(f"Successfully executed query: {state['query']}")

    except Exception as e:
        # Handle any errors during query execution
        error_message = f"Error executing query: {str(e)}"
        logger.error(error_message)
        state["results"] = None
        state["messages"] += [AIMessage(error_message)]

    return state


def extract_column_names(query: str) -> list:
    # Simple regex to extract column names from SELECT statement
    match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
    if match:
        columns = [col.strip() for col in match.group(1).split(',')]
        return [col.split()[-1] for col in columns]  # Handle cases like "MAX(column) AS max_column"
    return []


def preprocess_query_results(state: State) -> State:
    """
    Clean and preprocess the query results before visualization or analysis.

    Args:
        state (State): The current state containing query results.

    Returns:
        State: Updated state with preprocessed data.
    """
    if not state.get("results"):
        state["messages"].append(AIMessage("No results to preprocess."))
        return state

    try:
        # Extract column names from the query
        column_names = extract_column_names(state["query"])

        # Convert string results to a list of tuples
        results_list = ast.literal_eval(state["results"])

        # Convert results to a pandas DataFrame for easier manipulation
        df = pd.DataFrame(results_list, columns=column_names)

        # Basic cleaning steps
        df = df.dropna()  # Remove rows with missing values
        df = df.drop_duplicates()  # Remove duplicate rows

        # Convert date columns to datetime type
        date_columns = df.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError:
                pass  # Column couldn't be converted to datetime

        # Convert numeric columns to appropriate types
        df = df.apply(pd.to_numeric, errors='ignore')

        # Sort the DataFrame if there's a date column
        if 'date' in df.columns:
            df = df.sort_values('date')

        # Store the preprocessed data back in the state
        state["preprocessed_data"] = df.to_dict(orient='records')

        # Log the preprocessing steps
        preprocessing_summary = f"Preprocessed {len(df)} rows. Cleaned missing values and duplicates. Converted date and numeric columns."
        state["messages"].append(AIMessage(preprocessing_summary))

    except Exception as e:
        error_message = f"Error during data preprocessing: {str(e)}"
        state["messages"].append(AIMessage(error_message))
        state["preprocessed_data"] = None

    return state


def generate_visualization_and_narration(state: State) -> State:
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", REFLECTION_SYSTEM_PROMPT),
        ("system",
         "You are an AI assistant that generates visualizations and narrations based on data. Provide Plotly JSON for visualization and a narration explaining insights."),
        ("human", "Preprocessed data: {preprocessed_data}"),
        ("human",
         "Generate a visualization and narration. Use <visualization></visualization> tags for Plotly JSON and <narration></narration> tags for the narration.")
    ])

    response = llm.invoke(prompt.format(preprocessed_data=state["results"]))

    try:
        visualization_match = re.search(r'<visualization>(.*?)</visualization>', response.content, re.DOTALL)
        narration_match = re.search(r'<narration>(.*?)</narration>', response.content, re.DOTALL)

        if visualization_match and narration_match:
            visualization_data = json.loads(visualization_match.group(1))
            narration = narration_match.group(1).strip()

            # Validate Plotly JSON
            go.Figure(visualization_data)

            state["visualization_data"] = visualization_data
            state["narration"] = narration
        else:
            raise ValueError("Visualization or narration not found in the response")

    except Exception as e:
        error_message = f"Error generating visualization and narration: {str(e)}"
        logger.error(error_message)
        state["messages"].append(AIMessage(error_message))

    return state


def build_graph(toolkit) -> Any:
    graph = StateGraph(State)

    # Define nodes
    graph.add_node("retrieve_schema", lambda state: update_state_with_schema(state, toolkit))
    graph.add_node("decide_action", update_state_with_decision)
    graph.add_node("execute_query", lambda state: execute_query(state, toolkit))
    graph.add_node("preprocess_data", preprocess_query_results)
    graph.add_node("generate_output", generate_visualization_and_narration)

    # Define edges
    graph.add_edge("retrieve_schema", "decide_action")

    # Conditional edge from decide_action
    def route_action(state: State):
        if state.get("query"):
            return "execute_query"
        else:
            return END

    graph.add_edge("retrieve_schema", "decide_action")
    graph.add_conditional_edges("decide_action", route_action)
    graph.add_edge("execute_query", "preprocess_data")
    graph.add_edge("preprocess_data", "generate_output")
    graph.add_edge("generate_output", END)

    # Set the entrypoint
    graph.set_entry_point("retrieve_schema")

    return graph.compile()


@lru_cache(maxsize=100)
def cached_process_user_input(user_input: str):
    return process_user_input(user_input)


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)


class Input(BaseModel):
    input: str
    chat_history: List[Union[HumanMessage, AIMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )


class Output(BaseModel):
    output: Any


def inp(inpt: Input) -> dict:
    return {"messages": inpt["chat_history"] + [HumanMessage(inpt["input"])]}


def outp(outpt) -> str:
    return next(iter(outpt.values()))['messages'][-1].content


toolkit = create_sql_database_toolkit()

add_routes(
    app,
    (RunnableLambda(inp) | build_graph(toolkit) | RunnableLambda(outp)).with_types(input_type=Input,
                                                                                   output_type=Output).with_config(
        {"configurable": {"thread_id": uuid.uuid4()}}
    ),
)


def process_user_input(user_input: str):
    toolkit = create_sql_database_toolkit()
    graph = build_graph(toolkit)

    initial_state = State(
        messages=[HumanMessage(user_input)],
        schema=None,
        query=None,
        results=None,
        preprocessed_data=None,
        visualization_data=None,
        narration=None
    )

    try:
        for step in graph.stream(initial_state):
            current_node, current_state = list(step.items())[0]
            logger.info(f"Executing node: {current_node}")
            if current_node == "execute_query":
                current_state = execute_query(current_state, toolkit)

        final_state = list(step.values())[0]
        return final_state["visualization_data"], final_state["narration"]
    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        return None, f"An error occurred while processing your request: {str(e)}"


def test_decide_action():
    # Set up the initial state
    toolkit = create_sql_database_toolkit()
    schema_info = retrieve_schema(toolkit)

    # Test cases
    test_cases = [
        "Create a line chart showing the economic growth trends for all provinces from 2018 to 2023.",
        "What's the weather like today?",
        "List the top 5 customers by revenue",
    ]

    for case in test_cases:
        print(f"\nTest case: {case}")
        vis_data, narration = process_user_input(case)
        # state = State(
        #     messages=[HumanMessage(case)],
        #     schema=schema_info["schema"],
        #     query=None,
        #     results=None,
        #     visualization_data=None,
        #     narration=None
        # )
        # state = update_state_with_decision(state)
        # if state["query"] is not None:
        #     query = state["query"]
        #     print(f"SQL query: {query}")
        #     state = execute_query(state, toolkit)
        #     state = preprocess_query_results(state)
        #     state = generate_visualization_and_narration(state)
        # else:
        #     clarification = state["messages"][-1].content
        #     print(f"Clarification question: {clarification}")


if __name__ == "__main__":
    # test_decide_action()
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
