"""
Human-in-the-Loop Module for LangGraph

This module provides a production-grade implementation of a human-in-the-loop
workflow for LangGraph, allowing for manual approval of tool calls before execution.
"""

import json
import uuid
from typing import Any, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from typing_extensions import Annotated

from datanarrate.config import config as cfg


class State(TypedDict):
    """State definition for the graph."""
    messages: Annotated[List[Any], add_messages]
    tool_call_message: Optional[AIMessage]


class HumanInTheLoopAgent:
    """Human-in-the-Loop Agent implementation."""

    def __init__(self, tools: List[BaseTool], model: Optional[ChatOpenAI] = None):
        """Initialize the HumanInTheLoopAgent."""
        self.tools = tools
        self.model = model or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.model = self.model.bind_tools(tools)
        self.tool_executor = ToolExecutor(tools)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the StateGraph for the agent."""
        workflow = StateGraph(State)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("action", self._call_tool)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")
        return workflow.compile(checkpointer=self.memory)

    def _call_model(self, state: State) -> dict:
        """Call the language model and handle tool calls."""
        messages = state["messages"]
        if messages[-1].content == "y":
            return {
                "messages": [state["tool_call_message"]],
                "tool_call_message": None,
            }
        else:
            response = self.model.invoke(messages)
            if response.tool_calls:
                verification_message = self._generate_verification_message(response)
                response.id = str(uuid.uuid4())
                return {
                    "messages": [verification_message],
                    "tool_call_message": response,
                }
            else:
                return {
                    "messages": [response],
                    "tool_call_message": None,
                }

    def _call_tool(self, state: State) -> dict:
        """Execute the tool based on the last message."""
        messages = state["messages"]
        last_message = messages[-1]
        tool_call = last_message.tool_calls[0]
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        response = self.tool_executor.invoke(action)
        tool_message = ToolMessage(
            content=str(response), name=action.tool, tool_call_id=tool_call["id"]
        )
        return {"messages": [tool_message]}

    @staticmethod
    def _should_continue(state: State) -> str:
        """Determine whether to continue or end the conversation."""
        messages = state["messages"]
        last_message = messages[-1]
        return "continue" if last_message.tool_calls else "end"

    @staticmethod
    def _generate_verification_message(message: AIMessage) -> AIMessage:
        """Generate a verification message for tool calls."""
        serialized_tool_calls = json.dumps(message.tool_calls, indent=2)
        return AIMessage(
            content=(
                "I plan to invoke the following tools, do you approve?\n\n"
                "Type 'y' if you do, anything else to stop.\n\n"
                f"{serialized_tool_calls}"
            ),
            id=message.id,
        )

    def run(self, input_message: str, thread_id: str = "default") -> None:
        """Run the human-in-the-loop agent."""
        thread = {"configurable": {"thread_id": thread_id}}
        inputs = [HumanMessage(content=input_message)]

        for event in self.graph.stream({"messages": inputs}, thread, stream_mode="values"):
            event["messages"][-1].pretty_print()

    def continue_conversation(self, input_message: str, thread_id: str = "default") -> None:
        """Continue an existing conversation."""
        thread = {"configurable": {"thread_id": thread_id}}
        inputs = [HumanMessage(content=input_message)]

        for event in self.graph.stream({"messages": inputs}, thread, stream_mode="values"):
            event["messages"][-1].pretty_print()


# Usage example
if __name__ == "__main__":
    from langchain_core.tools import tool


    @tool
    def search(query: str):
        """Call to surf the web."""
        return ["It's sunny in San Francisco, but you better look out if you're a Gemini 😈."]


    llm = ChatOpenAI(model=cfg.LLM_MODEL_NAME, openai_api_base=cfg.OPENAI_API_BASE,
                     openai_api_key=cfg.OPENAI_API_KEY, temperature=0)

    agent = HumanInTheLoopAgent(tools=[search], model=llm)

    agent.run("What's the weather in SF?")
    agent.continue_conversation("Can you specify SF in CA?")
    agent.continue_conversation("y")
