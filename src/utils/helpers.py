import json
import logging
import re
from typing import Any, Dict, List, Union, Optional, Callable, Annotated

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""
    name: str = Field(..., description="The name of the tool being called")
    arguments: Dict[str, Any] = Field(..., description="The arguments passed to the tool")


class DataNarrationState(BaseModel):
    """Represents the state of the DataNarration system."""
    messages: Annotated[List[Union[AIMessage, HumanMessage]], add_messages]
    context: Dict[str, Any] = Field(default_factory=dict)
    task_plan: List[str] = Field(default_factory=list)
    current_step: int = 0
    output: str = ""


def parse_tool_calls(content: str) -> List[ToolCall]:
    """Parse tool calls from the content string."""
    pattern = r'(\w+)\((.*?)\)'
    matches = re.findall(pattern, content)
    return [ToolCall(name=name, arguments=dict(arg.split('=') for arg in args_str.split(','))) for name, args_str in
            matches]


async def execute_tool(tool: BaseTool, **kwargs) -> str:
    """Execute a tool with given arguments."""
    try:
        return await tool.arun(**kwargs) if hasattr(tool, 'arun') else tool.run(**kwargs)
    except Exception as e:
        logger.error(f"Error executing tool {tool.name}: {str(e)}", exc_info=True)
        raise


def get_last_message(messages: List[BaseMessage]) -> Optional[Union[AIMessage, HumanMessage]]:
    """Get the last AI or Human message from the list of messages."""
    return next((msg for msg in reversed(messages) if isinstance(msg, (AIMessage, HumanMessage))), None)


def update_dialog_stack(current_stack: List[str], update: str) -> List[str]:
    """Update the dialog stack based on the update string."""
    if update.startswith("+"):
        current_stack.append(update[1:])
    elif update.startswith("-") and current_stack and current_stack[-1] == update[1:]:
        current_stack.pop()
    return current_stack


def format_messages_for_prompt(messages: List[BaseMessage]) -> str:
    """Format a list of messages for inclusion in a prompt."""
    return "\n".join(f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in messages)


def truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate a string to a maximum length, adding an ellipsis if truncated."""
    return (s[:max_length] + '...') if len(s) > max_length else s


def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """Safely load a JSON string, returning an empty dict if parsing fails."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {json_str}")
        return {}


def create_tool_node_with_fallback(tools: List[BaseTool]) -> Dict:
    """Create a tool node with fallback handling."""

    def handle_tool_error(error: Exception) -> str:
        logger.error(f"Error occurred while executing tool: {str(error)}", exc_info=True)
        return f"Error occurred while executing tool: {str(error)}"

    return ToolNode(tools).with_fallbacks([handle_tool_error], exception_key="error")


def print_event(event: Dict[str, Any], printed: set, max_length: int = 1500):
    """Print an event, tracking which messages have been printed."""
    if current_state := event.get("dialog_state"):
        print(f"Currently in: {current_state[-1]}")
    if message := event.get("messages"):
        message = message[-1] if isinstance(message, list) else message
        if message.id not in printed:
            print(truncate_string(str(message), max_length))
            printed.add(message.id)


@StructuredTool.from_function
def generate_answer(answer: str) -> str:
    """Generate a final answer based on the analysis results."""
    return f"Based on the analysis, here's the answer: {answer}"


def create_state_graph(tools: List[BaseTool], llm: ChatOpenAI) -> StateGraph:
    """Create a StateGraph for the DataNarration system."""
    workflow = StateGraph(DataNarrationState)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant for data narration."),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    workflow.add_node("agent", agent_executor)
    workflow.add_node("tools", create_tool_node_with_fallback(tools))

    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


async def stream_results(graph: StateGraph, initial_state: DataNarrationState):
    """Stream the results of the DataNarration system."""
    async for event in graph.astream(initial_state):
        if event.event == "start":
            yield "Starting data narration process..."
        elif event.event == "end":
            yield f"Data narration complete. Final output: {event.state.output}"
        else:
            yield f"Executing {event.name}..."


def create_planner(llm: ChatOpenAI, tools: List[BaseTool], prompt: ChatPromptTemplate) -> Callable:
    """Create a planner for the DataNarration system."""
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


# Async versions of key functions
async def aparse_tool_calls(content: str) -> List[ToolCall]:
    return parse_tool_calls(content)


async def aget_last_message(messages: List[BaseMessage]) -> Optional[Union[AIMessage, HumanMessage]]:
    return get_last_message(messages)


async def aupdate_dialog_stack(current_stack: List[str], update: str) -> List[str]:
    return update_dialog_stack(current_stack, update)
