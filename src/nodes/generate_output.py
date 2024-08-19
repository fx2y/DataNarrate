import json
from typing import Dict, Any, List, Annotated

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from datanarrate.config import config


class OutputFormat(BaseModel):
    summary: str = Field(description="A concise summary of the analysis results")
    key_points: List[str] = Field(description="List of key points from the analysis")
    insights: List[str] = Field(description="List of insights derived from the analysis")
    visualizations: List[Dict[str, str]] = Field(
        description="List of visualization descriptions with type and description")
    next_steps: List[str] = Field(description="Suggested next steps or areas for further investigation")


class OutputState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    analysis_results: str
    output: OutputFormat = None


class OutputGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model_name=config.LLM_MODEL_NAME, openai_api_base=config.OPENAI_API_BASE,
                              openai_api_key=config.OPENAI_API_KEY, temperature=0.2)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert data analyst. Summarize the analysis results, provide key points, insights, and suggest relevant visualizations. Also recommend next steps for further investigation."),
            ("human",
             "Here are the analysis results:\n{analysis_results}\n\nGenerate a comprehensive output following the specified format.")
        ])
        self.chain = self.prompt | self.llm.with_structured_output(OutputFormat)

    async def agenerate(self, state: OutputState) -> Dict[str, Any]:
        try:
            output = await self.chain.ainvoke({"analysis_results": state.analysis_results})
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


async def generate_output(state: OutputState) -> Dict[str, Any]:
    generator = OutputGenerator()
    result = await generator.agenerate(state)
    return {
        "output": result["output"],
        "messages": [AIMessage(content=json.dumps(result["output"].dict(), indent=2))]
    }


def should_end(state: OutputState) -> str:
    if state.output:
        return END
    return "generate_output"


# Graph setup
workflow = StateGraph(OutputState)
workflow.add_node("generate_output", generate_output)
workflow.add_conditional_edges(
    "generate_output",
    should_end,
    {
        END: END,
        "generate_output": "generate_output"  # Allow for potential retries
    }
)

# Add any necessary tools
tools = []  # Add any tools that might be needed
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)


# Compile the graph
# app = workflow.compile()


# Example usage
async def run_workflow(input_data: str):
    initial_state = OutputState(
        messages=[HumanMessage(content=input_data)],
        analysis_results="Sample analysis results"
    )
    async for event in app.astream(initial_state):
        if "output" in event:
            print(json.dumps(event["output"].dict(), indent=2))
        elif "messages" in event:
            for message in event["messages"]:
                if isinstance(message, AIMessage):
                    print(f"AI: {message.content}")
                elif isinstance(message, HumanMessage):
                    print(f"Human: {message.content}")

# To run: await run_workflow("Analyze the data and provide insights")
