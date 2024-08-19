import logging
import os
from typing import List, Dict, Any, AsyncIterator

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ElasticsearchException, NotFoundError, RequestError
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Set up LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ElasticsearchTool"

logger = logging.getLogger(__name__)


class ElasticsearchQuery(BaseModel):
    """Schema for Elasticsearch query"""
    index: str = Field(..., description="The index to search in")
    query: Dict[str, Any] = Field(..., description="The Elasticsearch query")
    size: int = Field(10, description="Number of results to return")
    from_: int = Field(0, description="Starting offset for pagination")


class ElasticsearchResult(BaseModel):
    """Schema for Elasticsearch result"""
    hits: List[Dict[str, Any]] = Field(..., description="The search results")
    total: int = Field(..., description="Total number of matching documents")
    took: int = Field(..., description="Time in milliseconds for Elasticsearch to execute the search")


class ElasticsearchTool(BaseTool):
    """Tool for executing queries on Elasticsearch"""
    name: str = "elasticsearch"
    description: str = "Execute queries on Elasticsearch"
    args_schema: type[BaseModel] = ElasticsearchQuery
    return_direct: bool = False

    def __init__(self, es_client: AsyncElasticsearch):
        """
        Initialize the ElasticsearchTool.

        Args:
            es_client (AsyncElasticsearch): An initialized AsyncElasticsearch client.
        """
        super().__init__()
        self.es_client = es_client

    async def _arun(
            self,
            index: str,
            query: Dict[str, Any],
            size: int = 10,
            from_: int = 0,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> ElasticsearchResult:
        """
        Execute the Elasticsearch query.

        Args:
            index (str): The index to search in.
            query (Dict[str, Any]): The Elasticsearch query.
            size (int, optional): Number of results to return. Defaults to 10.
            from_ (int, optional): Starting offset for pagination. Defaults to 0.
            run_manager (Optional[CallbackManagerForToolRun]): Callback manager for the tool run.

        Returns:
            ElasticsearchResult: The search results.

        Raises:
            ValueError: If the Elasticsearch query fails.
        """
        try:
            if run_manager:
                await run_manager.on_tool_start(
                    {"name": self.name, "description": self.description},
                    {"index": index, "query": query, "size": size, "from_": from_},
                )
            response = await self.es_client.search(index=index, body=query, size=size, from_=from_)
            result = ElasticsearchResult(
                hits=[hit['_source'] for hit in response['hits']['hits']],
                total=response['hits']['total']['value'],
                took=response['took']
            )
            if run_manager:
                await run_manager.on_tool_end(str(result))
            return result
        except NotFoundError as e:
            error_msg = f"Index not found: {index}"
            logger.error(error_msg)
            if run_manager:
                await run_manager.on_tool_error(error_msg)
            raise ValueError(error_msg)
        except RequestError as e:
            error_msg = f"Invalid Elasticsearch query: {str(e)}"
            logger.error(error_msg)
            if run_manager:
                await run_manager.on_tool_error(error_msg)
            raise ValueError(error_msg)
        except ElasticsearchException as e:
            error_msg = f"Elasticsearch query failed: {str(e)}"
            logger.error(error_msg)
            if run_manager:
                await run_manager.on_tool_error(error_msg)
            raise ValueError(error_msg)

    async def arun(
            self,
            tool_input: Dict[str, Any],
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """
        Run the tool with the given input.

        Args:
            tool_input (Dict[str, Any]): The input for the tool.
            run_manager (Optional[CallbackManagerForToolRun]): Callback manager for the tool run.

        Returns:
            Dict[str, Any]: The search results.
        """
        query = ElasticsearchQuery(**tool_input)
        result = await self._arun(query.index, query.query, query.size, query.from_, run_manager)
        return result.dict()

    async def astream(
            self,
            tool_input: Dict[str, Any],
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream the tool execution with the given input.

        Args:
            tool_input (Dict[str, Any]): The input for the tool.
            run_manager (Optional[CallbackManagerForToolRun]): Callback manager for the tool run.

        Yields:
            Dict[str, Any]: Partial search results.
        """
        query = ElasticsearchQuery(**tool_input)
        result = await self._arun(query.index, query.query, query.size, query.from_, run_manager)

        # Simulate streaming by yielding partial results
        chunk_size = max(1, len(result.hits) // 5)
        for i in range(0, len(result.hits), chunk_size):
            partial_result = ElasticsearchResult(
                hits=result.hits[i:i + chunk_size],
                total=result.total,
                took=result.took
            )
            yield partial_result.dict()


async def arun_elasticsearch_tool(
        es_client: AsyncElasticsearch,
        index: str,
        query: Dict[str, Any],
        size: int = 10,
        from_: int = 0
) -> Dict[str, Any]:
    """
    Helper function to run the ElasticsearchTool.

    Args:
        es_client (AsyncElasticsearch): An initialized AsyncElasticsearch client.
        index (str): The index to search in.
        query (Dict[str, Any]): The Elasticsearch query.
        size (int, optional): Number of results to return. Defaults to 10.
        from_ (int, optional): Starting offset for pagination. Defaults to 0.

    Returns:
        Dict[str, Any]: The search results.
    """
    tool = ElasticsearchTool(es_client=es_client)
    return await tool.arun({"index": index, "query": query, "size": size, "from_": from_})


# Example usage in a LangGraph workflow
class State(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]
    current_step: str


async def elasticsearch_node(state: State, config: RunnableConfig):
    es_client = AsyncElasticsearch(hosts=["http://localhost:9200"])
    tool = ElasticsearchTool(es_client=es_client)
    tool_node = ToolNode(tools=[tool])

    result = await tool_node.ainvoke(state, config)
    return {**state, "messages": state["messages"] + [AIMessage(content=str(result))]}


def create_elasticsearch_graph():
    workflow = StateGraph(State)

    workflow.add_node("elasticsearch", elasticsearch_node)
    workflow.add_edge("elasticsearch", END)

    return workflow.compile()


# Example usage
async def main():
    graph = create_elasticsearch_graph()

    initial_state = State(
        messages=[HumanMessage(content="Search for documents about machine learning")],
        current_step="elasticsearch"
    )

    async for event in graph.astream(initial_state):
        if event.event == "start":
            print("Starting Elasticsearch query...")
        elif event.event == "end":
            print("Elasticsearch query complete.")
            print(event.state["messages"][-1].content)
        else:
            print(f"Executing {event.name}...")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
