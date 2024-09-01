from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import StructuredTool


def get_sql_tools(db, llm):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    return list_tables_tool, get_schema_tool


_DB_QUERY_DESCRIPTION = """
Execute a SQL query against the database and get back the result.
If the query is not correct, an error message will be returned.
If an error is returned, rewrite the query, check the query, and try again.
"""


def get_db_query_tool(db):
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def db_query_tool(query: str) -> str:
        """
        Execute a SQL query against the database and get back the result.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again.
        """
        result = db.run_no_throw(query)
        if not result:
            return "Error: Query failed. Please rewrite your query and try again."
        return result

    return StructuredTool.from_function(
        name="db_query_tool",
        func=db_query_tool,
        description=_DB_QUERY_DESCRIPTION
    )
