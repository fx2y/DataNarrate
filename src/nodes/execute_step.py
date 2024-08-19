from operator import add
from typing import Annotated, Dict, List, Any

from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from datanarrate.config import config


class ExecuteStepState(BaseModel):
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)
    intermediate_steps: Annotated[List[Dict], add] = Field(default_factory=list)
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)
    current_step: Any
    current_step_index: int


async def execute_step(state: ExecuteStepState, tools: List[BaseTool]) -> Dict:
    """
    Execute a single step in the task plan using the selected tools.

    This function processes the current step's tool calls, executes the appropriate tools,
    and returns the results as messages and intermediate steps to be added to the state.
    """
    tool_node = ToolNode(tools)

    current_step = state.current_step
    if not current_step or not current_step.tool:
        return {"messages": [], "intermediate_steps": [], "tool_outputs": {}}

    try:
        tool_name = current_step.tool.tool
        tool_args = current_step.tool.args.copy()

        # Resolve input dependencies
        for arg_name, arg_value in tool_args.items():
            if isinstance(arg_value, int):
                tool_args[arg_name] = state.tool_outputs.get(f"step_{arg_value}_output")
            elif isinstance(arg_value, list):
                tool_args[arg_name] = [
                    state.tool_outputs.get(f"step_{v}_output") if isinstance(v, int) else v
                    for v in arg_value
                ]

        # Handle LLM tool separately
        if tool_name.lower() == "llm":
            return {
                "messages": [],
                "intermediate_steps": [],
                "tool_outputs": {},
                "requires_reasoning": True,
                "reasoning_context": {
                    "current_step": state.current_step,
                    "tool_outputs": state.tool_outputs,
                }
            }

        tool_call = {
            "name": tool_name,
            "arguments": tool_args
        }

        result = await tool_node.ainvoke(
            {"messages": [FunctionMessage(content="", name=tool_name, additional_kwargs={"function_call": tool_call})]})
        tool_messages = result["messages"]

        intermediate_steps = [{
            "action": tool_call,
            "observation": tool_messages[0].content
        }]

        new_tool_outputs = {
            f"step_{state.current_step_index}_output": tool_messages[0].content
        }

        return {
            "messages": tool_messages,
            "intermediate_steps": intermediate_steps,
            "tool_outputs": new_tool_outputs,
            "requires_reasoning": False
        }
    except Exception as e:
        error_message = f"Error executing tool {tool_name}: {str(e)}"
        return {
            "messages": [FunctionMessage(content=error_message, name="error")],
            "intermediate_steps": [{
                "action": "error",
                "observation": error_message
            }],
            "tool_outputs": {},
            "requires_reasoning": False,
            "error_occurred": True
        }


# Example usage in a LangGraph context
if __name__ == "__main__":
    from langgraph.graph import StateGraph, END
    from langchain_core.tools import tool, BaseTool
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
