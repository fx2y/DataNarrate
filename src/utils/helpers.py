import json
import logging
import re
from typing import Any, Dict, List, Union, Optional

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""
    name: str = Field(..., description="The name of the tool being called")
    arguments: Dict[str, Any] = Field(..., description="The arguments passed to the tool")


def parse_tool_calls(content: str) -> List[ToolCall]:
    """
    Parse tool calls from the content string.

    Args:
        content (str): The string containing tool calls.

    Returns:
        List[ToolCall]: A list of parsed ToolCall objects.

    Example:
        >>> parse_tool_calls("search(query='LangChain'), calculate(x=5, y=3)")
        [ToolCall(name='search', arguments={'query': 'LangChain'}),
         ToolCall(name='calculate', arguments={'x': '5', 'y': '3'})]
    """
    pattern = r'(\w+)\((.*?)\)'
    matches = re.findall(pattern, content)
    tool_calls = []
    for name, args_str in matches:
        args = {}
        for arg in args_str.split(','):
            if '=' in arg:
                key, value = arg.split('=')
                args[key.strip()] = value.strip().strip('"')
        tool_calls.append(ToolCall(name=name, arguments=args))
    return tool_calls


async def execute_tool(tool: BaseTool, **kwargs) -> str:
    """
    Execute a tool with given arguments.

    Args:
        tool (BaseTool): The tool to execute.
        **kwargs: Arguments to pass to the tool.

    Returns:
        str: The result of the tool execution.

    Raises:
        Exception: If there's an error during tool execution.
    """
    try:
        if hasattr(tool, 'arun'):
            return await tool.arun(**kwargs)
        else:
            return tool.run(**kwargs)
    except Exception as e:
        logger.error(f"Error executing tool {tool.name}: {str(e)}", exc_info=True)
        raise


def get_last_message(messages: List[BaseMessage]) -> Optional[Union[AIMessage, HumanMessage]]:
    """
    Get the last AI or Human message from the list of messages.

    Args:
        messages (List[BaseMessage]): The list of messages to search.

    Returns:
        Optional[Union[AIMessage, HumanMessage]]: The last AI or Human message, or None if not found.
    """
    for message in reversed(messages):
        if isinstance(message, (AIMessage, HumanMessage)):
            return message
    return None


def update_dialog_stack(current_stack: List[str], update: str) -> List[str]:
    """
    Update the dialog stack based on the update string.

    Args:
        current_stack (List[str]): The current dialog stack.
        update (str): The update string, starting with '+' to push or '-' to pop.

    Returns:
        List[str]: The updated dialog stack.

    Example:
        >>> update_dialog_stack(['main'], '+sub_dialog')
        ['main', 'sub_dialog']
        >>> update_dialog_stack(['main', 'sub_dialog'], '-sub_dialog')
        ['main']
    """
    if update.startswith("+"):
        current_stack.append(update[1:])
    elif update.startswith("-"):
        if current_stack and current_stack[-1] == update[1:]:
            current_stack.pop()
    return current_stack


def format_messages_for_prompt(messages: List[BaseMessage]) -> str:
    """
    Format a list of messages for inclusion in a prompt.

    Args:
        messages (List[BaseMessage]): The list of messages to format.

    Returns:
        str: A formatted string representation of the messages.
    """
    return "\n".join(
        f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in messages
    )


def truncate_string(s: str, max_length: int = 100) -> str:
    """
    Truncate a string to a maximum length, adding an ellipsis if truncated.

    Args:
        s (str): The string to truncate.
        max_length (int, optional): The maximum length of the string. Defaults to 100.

    Returns:
        str: The truncated string.
    """
    return (s[:max_length] + '...') if len(s) > max_length else s


def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """
    Safely load a JSON string, returning an empty dict if parsing fails.

    Args:
        json_str (str): The JSON string to parse.

    Returns:
        Dict[str, Any]: The parsed JSON as a dictionary, or an empty dict if parsing fails.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {json_str}")
        return {}


def create_tool_node_with_fallback(tools: List[BaseTool]) -> Dict:
    """
    Create a tool node with fallback handling.

    Args:
        tools (List[BaseTool]): The list of tools to include in the node.

    Returns:
        Dict: A dictionary representing the tool node with fallback handling.
    """
    from langgraph.prebuilt import ToolNode

    def handle_tool_error(error: Exception) -> str:
        logger.error(f"Error occurred while executing tool: {str(error)}", exc_info=True)
        return f"Error occurred while executing tool: {str(error)}"

    return ToolNode(tools).with_fallbacks(
        [handle_tool_error], exception_key="error"
    )


def print_event(event: Dict[str, Any], printed: set, max_length: int = 1500):
    """
    Print an event, tracking which messages have been printed.

    Args:
        event (Dict[str, Any]): The event to print.
        printed (set): A set of message IDs that have already been printed.
        max_length (int, optional): The maximum length of the printed message. Defaults to 1500.
    """
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: {current_state[-1]}")

    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in printed:
            msg_repr = truncate_string(str(message), max_length)
            print(msg_repr)
            printed.add(message.id)


# Async versions of some functions for better integration with LangGraph

async def aparse_tool_calls(content: str) -> List[ToolCall]:
    """Async version of parse_tool_calls"""
    return parse_tool_calls(content)


async def aget_last_message(messages: List[BaseMessage]) -> Optional[Union[AIMessage, HumanMessage]]:
    """Async version of get_last_message"""
    return get_last_message(messages)


async def aupdate_dialog_stack(current_stack: List[str], update: str) -> List[str]:
    """Async version of update_dialog_stack"""
    return update_dialog_stack(current_stack, update)
