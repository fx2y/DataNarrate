import logging
from functools import lru_cache
from typing import Dict, Any, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, Node
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    """Structured output for query analysis"""
    task_type: str = Field(description="The type of task required (e.g., data analysis, visualization, explanation)")
    sub_tasks: List[str] = Field(description="List of sub-tasks needed to complete the main task")
    required_data_sources: List[str] = Field(description="List of data sources needed to answer the query")
    constraints: List[str] = Field(description="Any constraints or specific requirements mentioned in the query")
    potential_insights: List[str] = Field(description="Potential insights or angles to explore based on the query")


class AnalyzeQueryNode(Node):
    def __init__(self):
        self.output_parser = StructuredOutputParser.from_pydantic_model(QueryAnalysis)
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert data analyst assistant. Analyze the following user query and provide a structured output:

            User Query: {query}

            {format_instructions}

            Provide a detailed analysis of the query, breaking it down into its components and identifying key aspects for processing.
            """
        )
        self.model = ChatAnthropic(model="claude-3-haiku-20240307")

    @lru_cache(maxsize=100)
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a query and cache the result"""
        input_dict = {
            "query": query,
            "format_instructions": self.output_parser.get_format_instructions()
        }
        response = self.model.invoke(self.prompt.format_prompt(**input_dict))
        return self.output_parser.parse(response.content)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes the user's query and updates the state with structured analysis."""
        try:
            # Extract the latest user message
            user_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)

            if not user_message:
                logger.warning("No user message found in the state")
                return state  # No user message found, return unchanged state

            # Generate the analysis
            analysis = self._analyze_query(user_message.content)

            # Create AI message with the analysis
            ai_message = AIMessage(content=str(analysis))

            # Return new state with the analysis and updated messages
            return {
                **state,
                "query_analysis": analysis.dict(),
                "messages": state["messages"] + [ai_message]
            }
        except Exception as e:
            logger.error(f"Error in analyze_query: {str(e)}")
            return {
                **state,
                "error": f"Failed to analyze query: {str(e)}"
            }


# Create the graph
def create_analyze_query_graph() -> CompiledStateGraph:
    workflow = StateGraph()
    workflow.add_node("analyze_query", AnalyzeQueryNode())
    workflow.set_entry_point("analyze_query")
    return workflow.compile()


# Usage
analyze_query_graph = create_analyze_query_graph()


async def analyze_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper function to run the analyze_query graph"""
    return await analyze_query_graph.arun(state)
