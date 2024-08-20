from typing import Annotated, Dict, List, Any, Union

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from datanarrate.config import config


class ExecuteStepState(BaseModel):
    messages: Annotated[List[BaseMessage], Field(default_factory=list)]
    intermediate_steps: Annotated[List[Dict], Field(default_factory=list)]
    tool_outputs: Dict[str, Union[str, List[str]]] = Field(default_factory=dict)
    current_step: Dict[str, str]
    current_step_index: int


class StepExecutor:
    def __init__(self, tools: List[BaseTool]):
        self.tool_node = ToolNode(tools)
        self.llm = ChatOpenAI(
            model_name=config.LLM_MODEL_NAME,
            openai_api_base=config.OPENAI_API_BASE,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.2
        )
        self.llm_with_tools = self.llm.bind_tools(tools)

    async def execute_step(self, state: ExecuteStepState) -> Dict[str, Any]:
        if not state.current_step:
            return {
                "messages": [],
                "intermediate_steps": [],
                "tool_outputs": {},
                "requires_reasoning": False,
                "reasoning_context": None
            }

        try:
            # Use the LLM to generate tool calls based on the current step
            llm_response = await self.llm_with_tools.ainvoke([
                AIMessage(content=f"Execute the following step: {state.current_step['description']}")
            ])

            if not llm_response.tool_calls:
                return {
                    "messages": [llm_response],
                    "intermediate_steps": [],
                    "tool_outputs": {},
                    "requires_reasoning": True,
                    "reasoning_context": {
                        "current_step": state.current_step,
                        "tool_outputs": state.tool_outputs,
                    }
                }

            # Use ToolNode to execute the tool calls
            tool_node_input = {"messages": [llm_response]}
            tool_node_output = await self.tool_node.ainvoke(tool_node_input)

            # Process the tool outputs
            new_messages = tool_node_output["messages"]
            intermediate_steps = []
            new_tool_outputs = {}

            for i, (tool_call, tool_message) in enumerate(zip(llm_response.tool_calls, new_messages)):
                intermediate_steps.append({
                    "action": {
                        "tool": tool_call.function.name,
                        "tool_input": tool_call.function.arguments,
                    },
                    "observation": tool_message.content,
                })
                new_tool_outputs[f"step_{state.current_step_index}_output_{i}"] = tool_message.content

            return {
                "messages": new_messages,
                "intermediate_steps": intermediate_steps,
                "tool_outputs": new_tool_outputs,
                "requires_reasoning": False,
                "reasoning_context": None
            }

        except Exception as e:
            error_message = f"Error executing step: {str(e)}"
            return {
                "messages": [AIMessage(content=error_message)],
                "intermediate_steps": [{
                    "action": "error",
                    "observation": error_message,
                }],
                "tool_outputs": {},
                "requires_reasoning": False,
                "reasoning_context": None,
                "error_occurred": True
            }


async def execute_step(state: ExecuteStepState, tools: List[BaseTool]) -> Dict:
    """
    Execute a single step in the task plan using the selected tools.

    This function processes the current step's tool calls, executes the appropriate tools,
    and returns the results as messages and intermediate steps to be added to the state.
    """
    executor = StepExecutor(tools)
    return await executor.execute_step(state)


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
