import json

from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler

from datanarrate.config import config as cfg
from sql_agent.config import initialize_mysql_database
from sql_agent.workflow import create_workflow

langfuse_handler = CallbackHandler()


def run_sql_agent(question: str, app):
    messages = app.invoke({"messages": [("user", question)]}, config={"callbacks": [langfuse_handler]})
    json_str = messages["messages"][-1].additional_kwargs["tool_calls"][0]["function"]["arguments"]
    return json.loads(json_str)["final_answer"]


if __name__ == '__main__':
    llm = ChatOpenAI(model=cfg.LLM_MODEL_NAME, openai_api_base=cfg.OPENAI_API_BASE,
                     openai_api_key=cfg.OPENAI_API_KEY, temperature=0)

    try:
        db, toolkit = initialize_mysql_database(
            host=cfg.MYSQL_HOST,
            user=cfg.MYSQL_USER,
            password=cfg.MYSQL_PASSWORD,
            database=cfg.MYSQL_DATABASE,
            llm=llm
        )

        app = create_workflow(llm, db)

        question = "Which sales agent made the most in sales in 2009?"
        answer = run_sql_agent(question, app)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error initializing database: {e}")
        # Handle the error appropriately
