import json
from typing import Dict, Any, List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, State
from pydantic import BaseModel, Field


class OutputFormat(BaseModel):
    summary: str = Field(description="A concise summary of the analysis results")
    key_points: List[str] = Field(description="List of key points from the analysis")
    insights: List[str] = Field(description="List of insights derived from the analysis")
    visualizations: List[Dict[str, str]] = Field(
        description="List of visualization descriptions with type and description")
    next_steps: List[str] = Field(description="Suggested next steps or areas for further investigation")


class OutputState(State):
    analysis_results: str
    output: OutputFormat = None


class OutputGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert data analyst. Summarize the analysis results, provide key points, insights, and suggest relevant visualizations. Also recommend next steps for further investigation."),
            ("human",
             "Here are the analysis results:\n{analysis_results}\n\nGenerate a comprehensive output following the specified format.")
        ])
        self.chain = self.prompt | self.llm.with_structured_output(OutputFormat)

    async def agenerate(self, state: OutputState, config: RunnableConfig) -> Dict[str, Any]:
        try:
            output = await self.chain.ainvoke({"analysis_results": state.analysis_results}, config)
            return {"output": output}
        except Exception as e:
            print(f"Error in output generation: {str(e)}")
            return {"output": OutputFormat(
                summary="Error occurred during output generation.",
                key_points=["Error in processing"],
                insights=[],
                visualizations=[],
                next_steps=["Retry analysis"]
            )}


async def generate_output(state: OutputState, config: RunnableConfig) -> Dict[str, Any]:
    generator = OutputGenerator()
    result = await generator.agenerate(state, config)
    return {
        "output": result["output"],
        "messages": [AIMessage(content=json.dumps(result["output"].dict(), indent=2))]
    }


# Graph setup
workflow = StateGraph(OutputState)
workflow.add_node("generate_output", generate_output)
# Add more nodes and edges as needed
app = workflow.compile()
