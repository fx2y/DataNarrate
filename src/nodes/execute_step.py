from operator import add
from typing import Annotated, Dict, List

from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from datanarrate.config import config


class ExecuteStepState(BaseModel):
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)
    intermediate_steps: Annotated[List[Dict], add] = Field(default_factory=list)


async def execute_step(state: ExecuteStepState, tools: List[BaseTool]) -> Dict:
    """
    Execute a single step in the task plan using the selected tools.

    This function processes the last message's tool calls, executes the appropriate tools,
    and returns the results as messages and intermediate steps to be added to the state.
    """
    tool_node = ToolNode(tools)

    last_message = state.messages[-1]
    if not last_message.additional_kwargs.get("tool_calls"):
        return {"messages": [], "intermediate_steps": []}

    try:
        result = await tool_node.ainvoke({"messages": [last_message]})
        tool_messages = result["messages"]

        intermediate_steps = []
        for tool_call, tool_message in zip(last_message.additional_kwargs["tool_calls"], tool_messages):
            intermediate_steps.append({
                "action": tool_call,
                "observation": tool_message.content
            })

        return {
            "messages": tool_messages,
            "intermediate_steps": intermediate_steps
        }
    except Exception as e:
        error_message = f"Error executing tools: {str(e)}"
        return {
            "messages": [FunctionMessage(content=error_message, name="error")],
            "intermediate_steps": [{
                "action": "error",
                "observation": error_message
            }]
        }


# Example usage in a LangGraph context
if __name__ == "__main__":
    from langgraph.graph import StateGraph, END
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage


    @tool
    def search(query: str) -> str:
        """Search the web for information."""
        return f"Search results for: {query}"


    workflow = StateGraph(ExecuteStepState)

    workflow.add_node("execute_step", lambda state: execute_step(state, tools=[search]))

    workflow.add_edge("execute_step", END)

    app = workflow.compile()

    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.2
    )
    llm_with_tools = llm.bind_tools([search])

    initial_state = ExecuteStepState(
        messages=[
            HumanMessage(content="Find information about LangGraph"),
            llm_with_tools.invoke([HumanMessage(content="Find information about LangGraph")])
        ]
    )

    result = app.invoke(initial_state)
    print("Messages:", result["messages"])
    print("Intermediate Steps:", result["intermediate_steps"])
