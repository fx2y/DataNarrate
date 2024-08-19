import logging
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from datanarrate.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceInfo(BaseModel):
    name: str = Field(description="Name of the data source (e.g., 'mysql' or 'elasticsearch')")
    relevant_tables_or_indices: List[str] = Field(description="List of relevant tables or indices for this data source")
    suggested_fields: Dict[str, List[str]] = Field(description="Suggested fields for each table or index")


class QueryAnalysis(BaseModel):
    """Structured output for query analysis"""
    task_type: str = Field(description="The type of task required (e.g., data analysis, visualization, explanation)")
    sub_tasks: List[str] = Field(description="List of sub-tasks needed to complete the main task")
    required_data_sources: List[DataSourceInfo] = Field(
        description="List of required data sources with their relevant tables/indices and fields")
    constraints: List[str] = Field(description="Any constraints or specific requirements mentioned in the query")
    potential_insights: List[str] = Field(description="Potential insights or angles to explore based on the query")


class AnalyzeQueryNode:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert data analyst assistant. Analyze the following user query and provide a structured output:

            User Query: {query}

            Unified Compressed Schema:
            {schema}

            Provide a detailed analysis of the query, breaking it down into its components and identifying key aspects for processing.
            Use the provided schema to identify relevant data sources, tables/indices, and fields.

            Output the result as a JSON object with the following structure:
            {{
                "task_type": "string",
                "sub_tasks": ["string"],
                "required_data_sources": [
                    {{
                        "name": "string",
                        "relevant_tables_or_indices": ["string"],
                        "suggested_fields": {{"table_name": ["field1", "field2"]}}
                    }}
                ],
                "constraints": ["string"],
                "potential_insights": ["string"]
            }}
            """
        )
        self.model = ChatOpenAI(
            model_name=config.LLM_MODEL_NAME,
            openai_api_base=config.OPENAI_API_BASE,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.2
        )
        self.chain = self.prompt | self.model.with_structured_output(QueryAnalysis)

    async def run(self, state: Dict[str, Any], config) -> Dict[str, Any]:
        """Analyzes the user's query and updates the state with structured analysis."""
        try:
            # Extract the latest user message
            user_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)

            if not user_message:
                logger.warning("No user message found in the state")
                return state  # No user message found, return unchanged state

            # Generate the analysis
            analysis = self.chain.invoke({
                "query": user_message.content,
                "schema": config["configurable"].get("schema_info", {})
            })

            # Create AI message with the analysis
            ai_message = AIMessage(content=str(analysis))

            # Return new state with the analysis and updated messages
            return {
                "query_analysis": analysis.dict(),
                "messages": state["messages"] + [ai_message]
            }
        except Exception as e:
            logger.error(f"Error in analyze_query: {str(e)}")
            return {
                "error": f"Failed to analyze query: {str(e)}"
            }


# Create the graph
def create_analyze_query_graph() -> CompiledStateGraph:
    analyze_query_node = AnalyzeQueryNode()
    workflow = StateGraph(Dict[str, Any])
    workflow.add_node("analyze_query", analyze_query_node.run)
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", END)
    return workflow.compile()


# Usage
analyze_query_graph = create_analyze_query_graph()


async def analyze_query(state: Dict[str, Any], config) -> Dict[str, Any]:
    """Analyzes the user's query and updates the state with structured analysis."""
    try:
        node = AnalyzeQueryNode()
        result = await node.run(state, config)
        return {
            **state,
            "query_analysis": result["query_analysis"],
            "messages": result["messages"]
        }
    except Exception as e:
        logger.error(f"Error in analyze_query: {str(e)}")
        return {
            **state,
            "error": f"Failed to analyze query: {str(e)}"
        }
