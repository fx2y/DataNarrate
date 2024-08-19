import logging
from typing import List, Dict, Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ElasticsearchException, NotFoundError, RequestError
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

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
            from_: int = 0
    ) -> ElasticsearchResult:
        """
        Execute the Elasticsearch query.

        Args:
            index (str): The index to search in.
            query (Dict[str, Any]): The Elasticsearch query.
            size (int, optional): Number of results to return. Defaults to 10.
            from_ (int, optional): Starting offset for pagination. Defaults to 0.

        Returns:
            ElasticsearchResult: The search results.

        Raises:
            ValueError: If the Elasticsearch query fails.
        """
        try:
            response = await self.es_client.search(index=index, body=query, size=size, from_=from_)
            return ElasticsearchResult(
                hits=[hit['_source'] for hit in response['hits']['hits']],
                total=response['hits']['total']['value'],
                took=response['took']
            )
        except NotFoundError as e:
            logger.error(f"Index not found: {index}")
            raise ValueError(f"Index not found: {index}")
        except RequestError as e:
            logger.error(f"Invalid query: {query}")
            raise ValueError(f"Invalid Elasticsearch query: {str(e)}")
        except ElasticsearchException as e:
            logger.error(f"Elasticsearch query failed: {str(e)}")
            raise ValueError(f"Elasticsearch query failed: {str(e)}")

    async def arun(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the tool with the given input.

        Args:
            tool_input (Dict[str, Any]): The input for the tool.

        Returns:
            Dict[str, Any]: The search results.
        """
        query = ElasticsearchQuery(**tool_input)
        result = await self._arun(query.index, query.query, query.size, query.from_)
        return result.dict()


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
