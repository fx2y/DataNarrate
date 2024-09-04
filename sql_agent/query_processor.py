from typing import Dict, Union, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from datanarrate.config import config as cfg


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class RequestClarification(BaseModel):
    """Request clarification from the user about their query."""
    question: str = Field(description="The clarification question to ask the user")


class GenerateSQLQuery(BaseModel):
    """Generate a SQL query based on the user's request and database schema."""
    query: str = Field(description="The SQL query to execute")


class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                    not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [HumanMessage(content="Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": [result]}


def process_user_request(user_request: str, db_schema: Dict[str, list]) -> Dict[str, Union[str, bool]]:
    llm = ChatOpenAI(model=cfg.LLM_MODEL_NAME, openai_api_base=cfg.OPENAI_API_BASE,
                     openai_api_key=cfg.OPENAI_API_KEY, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI assistant that helps with database queries. Given a user's request and a database schema, decide whether to ask for clarification or craft a SQL query."),
        ("placeholder", "{messages}")
    ])

    tools = [RequestClarification, GenerateSQLQuery]
    runnable = prompt | llm.bind_tools(tools)

    assistant = Assistant(runnable)

    workflow = StateGraph(State)

    def initial_state(state):
        return {
            "messages": [
                SystemMessage(content="You are a helpful database query assistant."),
                HumanMessage(content=f"User request: {user_request}\n\nDatabase schema: {db_schema}")
            ]
        }

    workflow.add_node("assistant", assistant)

    workflow.set_entry_point("assistant")
    workflow.add_conditional_edges(
        "assistant",
        lambda x: END if x["messages"][-1].tool_calls[0]["name"] == GenerateSQLQuery.__name__ else "assistant"
    )

    # memory = MemorySaver()
    # app = workflow.compile(checkpointer=memory)
    app = workflow.compile()
    result = app.invoke(initial_state({}))
    last_message = result["messages"][-1]

    if last_message.tool_calls[0]["name"] == RequestClarification.__name__:
        return {
            "needs_clarification": True,
            "clarification_question": last_message.tool_calls[0]["args"]["question"]
        }
    else:
        return {
            "needs_clarification": False,
            "sql_query": last_message.tool_calls[0]["args"]["query"]
        }


if __name__ == '__main__':
    # Example usage
    user_request = "Show me the total sales for each product category"
    db_schema = {
        "products": ["id", "name", "category", "price"],
        "sales": ["id", "product_id", "quantity", "date"]
    }

    result = process_user_request(user_request, db_schema)
    print(result)
