import re
from typing import Any, Dict, List, Optional, Annotated

import pandas as pd
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.ext.asyncio import create_async_engine


class MySQLConfig(BaseModel):
    host: str = Field(..., description="MySQL host")
    port: int = Field(..., description="MySQL port")
    user: str = Field(..., description="MySQL user")
    password: str = Field(..., description="MySQL password")
    database: str = Field(..., description="MySQL database name")

    @validator('port')
    def port_must_be_valid(cls, v):
        if v < 0 or v > 65535:
            raise ValueError('Port must be between 0 and 65535')
        return v


class MySQLTool(StructuredTool):
    name: str = "mysql_tool"
    description: str = "Execute SQL queries on a MySQL database, list tables, or describe table schemas"
    config: MySQLConfig = Field(..., description="MySQL connection configuration")
    max_rows: int = Field(1000, description="Maximum number of rows to return")

    def _get_connection_string(self, async_conn: bool = False) -> str:
        dialect = "mysql+aiomysql" if async_conn else "mysql+pymysql"
        return f"{dialect}://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"

    def _sanitize_query(self, query: str) -> str:
        query = re.sub(r'/\*.*?\*/', '', query)
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        query = re.sub(r'\s+', ' ', query).strip()
        return query

    def _execute_query(self, query: str) -> pd.DataFrame:
        engine = create_engine(self._get_connection_string())
        with engine.connect() as connection:
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchmany(self.max_rows), columns=result.keys())
            if len(df) == self.max_rows:
                print(f"Warning: Result set truncated to {self.max_rows} rows")
            return df

    async def _execute_query_async(self, query: str) -> pd.DataFrame:
        engine = create_async_engine(self._get_connection_string(async_conn=True))
        async with engine.connect() as connection:
            result = await connection.execute(text(query))
            df = pd.DataFrame(await result.fetchmany(self.max_rows), columns=result.keys())
            if len(df) == self.max_rows:
                print(f"Warning: Result set truncated to {self.max_rows} rows")
            return df

    def _list_tables(self) -> List[str]:
        engine = create_engine(self._get_connection_string())
        inspector = inspect(engine)
        return inspector.get_table_names()

    def _describe_table(self, table_name: str) -> List[Dict[str, Any]]:
        engine = create_engine(self._get_connection_string())
        inspector = inspect(engine)
        return [{'name': col['name'], 'type': str(col['type'])} for col in inspector.get_columns(table_name)]

    def _run(
            self,
            query: str,
            state: Annotated[Dict[str, Any], InjectedState] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            if query.lower().startswith("list tables"):
                return f"Available tables: {', '.join(self._list_tables())}"
            elif query.lower().startswith("describe table"):
                table_name = query.split()[-1]
                return f"Schema for {table_name}: {self._describe_table(table_name)}"
            else:
                sanitized_query = self._sanitize_query(query)
                result_df = self._execute_query(sanitized_query)
                return result_df.to_string(index=False)
        except Exception as e:
            return f"Error executing query: {str(e)}\nQuery: {query}"

    async def _arun(
            self,
            query: str,
            state: Annotated[Dict[str, Any], InjectedState] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            if query.lower().startswith("list tables"):
                return f"Available tables: {', '.join(self._list_tables())}"
            elif query.lower().startswith("describe table"):
                table_name = query.split()[-1]
                return f"Schema for {table_name}: {self._describe_table(table_name)}"
            else:
                sanitized_query = self._sanitize_query(query)
                result_df = await self._execute_query_async(sanitized_query)
                return result_df.to_string(index=False)
        except Exception as e:
            return f"Error executing query: {str(e)}\nQuery: {query}"


def create_mysql_tool(config: MySQLConfig, max_rows: int = 1000) -> MySQLTool:
    return MySQLTool(config=config, max_rows=max_rows)
