import ast
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
ID_PATTERN = r"\$\{?(\d+)\}?"
END_OF_PLAN = "<END_OF_PLAN>"


### Helper functions

def _ast_parse(arg: str) -> Any:
    """Parse a string argument using AST literal evaluation.

    Args:
        arg (str): The string argument to parse.

    Returns:
        Any: The parsed value.

    Raises:
        ValueError: If the argument cannot be parsed.
    """
    try:
        return ast.literal_eval(arg)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse argument '{arg}': {e}")


def _parse_llm_compiler_action_args(args: str, tool: Union[str, BaseTool]) -> Dict[str, Any]:
    """Parse arguments from a string for a given tool.

    Args:
        args (str): The string containing the arguments.
        tool (Union[str, BaseTool]): The tool for which to parse the arguments.

    Returns:
        Dict[str, Any]: The parsed arguments.
    """
    if not args or isinstance(tool, str):
        return {}

    extracted_args = {}
    tool_key = None
    prev_idx = None

    for key in tool.args.keys():
        if f"{key}=" in args:
            idx = args.index(f"{key}=")
            if prev_idx is not None:
                extracted_args[tool_key] = _ast_parse(args[prev_idx:idx].strip().rstrip(","))
            args = args.split(f"{key}=", 1)[1]
            tool_key = key
            prev_idx = 0

    if prev_idx is not None:
        extracted_args[tool_key] = _ast_parse(args[prev_idx:].strip().rstrip(",").rstrip(")"))

    return extracted_args


def default_dependency_rule(idx: int, args: str) -> bool:
    """Default rule to determine dependencies based on argument patterns.

    Args:
        idx (int): The index to check for dependencies.
        args (str): The arguments to check for dependencies.

    Returns:
        bool: True if the index is a dependency, False otherwise.
    """
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


def _get_dependencies_from_graph(idx: int, tool_name: str, args: Dict[str, Any]) -> List[int]:
    """Get dependencies from a graph based on tool name and arguments.

    Args:
        idx (int): The index of the current task.
        tool_name (str): The name of the tool.
        args (Dict[str, Any]): The arguments for the tool.

    Returns:
        List[int]: A list of dependencies.
    """
    if tool_name == "join":
        return list(range(1, idx))
    return [i for i in range(1, idx) if default_dependency_rule(i, str(args))]


class Task(TypedDict):
    idx: int
    tool: Union[str, BaseTool]
    args: Dict[str, Any]
    dependencies: List[int]
    thought: Optional[str]


def instantiate_task(
        tools: Sequence[BaseTool],
        idx: int,
        tool_name: str,
        args: Union[str, Any],
        thought: Optional[str] = None,
) -> Task:
    """Instantiate a task with given tools, index, tool name, arguments, and thought.

    Args:
        tools (Sequence[BaseTool]): The available tools.
        idx (int): The index of the task.
        tool_name (str): The name of the tool.
        args (Union[str, Any]): The arguments for the tool.
        thought (Optional[str]): The thought associated with the task.

    Returns:
        Task: The instantiated task.

    Raises:
        OutputParserException: If the tool is not found.
    """
    if tool_name == "join":
        tool = "join"
    else:
        try:
            tool = tools[[tool.name for tool in tools].index(tool_name)]
        except ValueError as e:
            raise OutputParserException(f"Tool {tool_name} not found.") from e

    tool_args = _parse_llm_compiler_action_args(args, tool)
    dependencies = _get_dependencies_from_graph(idx, tool_name, tool_args)

    return Task(
        idx=idx,
        tool=tool,
        args=tool_args,
        dependencies=dependencies,
        thought=thought,
    )


class LLMCompilerPlanParser(BaseTransformOutputParser[Dict[str, Any]], extra="allow"):
    """Parser for transforming LLM output into tasks."""

    tools: List[BaseTool]

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
        """Transform input into tasks.

        Args:
            input (Iterator[Union[str, BaseMessage]]): The input to transform.

        Yields:
            Iterator[Task]: The transformed tasks.
        """
        texts = []
        thought = None
        for chunk in input:
            text = chunk if isinstance(chunk, str) else str(chunk.content)
            for task, thought in self.ingest_token(text, texts, thought):
                yield task
        if texts:
            task, _ = self._parse_task("".join(texts), thought)
            if task:
                yield task

    def parse(self, text: str) -> List[Task]:
        """Parse a single text input into a list of tasks.

        Args:
            text (str): The text to parse.

        Returns:
            List[Task]: The parsed tasks.
        """
        return list(self._transform([text]))

    def stream(
            self,
            input: Union[str, BaseMessage],
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Iterator[Task]:
        """Stream input and transform it into tasks.

        Args:
            input (Union[str, BaseMessage]): The input to stream.
            config (Optional[RunnableConfig]): The configuration for the stream.
            **kwargs (Optional[Any]): Additional arguments.

        Yields:
            Iterator[Task]: The transformed tasks.
        """
        yield from self.transform([input], config, **kwargs)

    def ingest_token(
            self, token: str, buffer: List[str], thought: Optional[str]
    ) -> Iterator[Tuple[Optional[Task], str]]:
        """Ingest a token and update the buffer and thought.

        Args:
            token (str): The token to ingest.
            buffer (List[str]): The buffer to update.
            thought (Optional[str]): The current thought.

        Yields:
            Iterator[Tuple[Optional[Task], str]]: The updated task and thought.
        """
        buffer.append(token)
        if "\n" in token:
            buffer_ = "".join(buffer).split("\n")
            suffix = buffer_[-1]
            for line in buffer_[:-1]:
                task, thought = self._parse_task(line, thought)
                if task:
                    yield task, thought
            buffer.clear()
            buffer.append(suffix)

    def _parse_task(self, line: str, thought: Optional[str] = None) -> Tuple[Optional[Task], Optional[str]]:
        """Parse a single line into a task.

        Args:
            line (str): The line to parse.
            thought (Optional[str]): The current thought.

        Returns:
            Tuple[Optional[Task], Optional[str]]: The parsed task and updated thought.
        """
        task = None
        if match := re.match(THOUGHT_PATTERN, line):
            thought = match.group(1)
        elif match := re.match(ACTION_PATTERN, line):
            idx, tool_name, args, _ = match.groups()
            idx = int(idx)
            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )
            thought = None
        return task, thought


### Unit Tests

import unittest


class TestLLMCompilerPlanParser(unittest.TestCase):

    def setUp(self):
        self.tools = [BaseTool(name="tool1", args={"arg1": str}), BaseTool(name="tool2", args={"arg2": int})]
        self.parser = LLMCompilerPlanParser(tools=self.tools)

    def test_ast_parse(self):
        self.assertEqual(_ast_parse("1"), 1)
        self.assertEqual(_ast_parse("'string'"), "string")
        with self.assertRaises(ValueError):
            _ast_parse("invalid")

    def test_parse_llm_compiler_action_args(self):
        args = "arg1='value1', arg2=2"
        tool = self.tools[0]
        parsed_args = _parse_llm_compiler_action_args(args, tool)
        self.assertEqual(parsed_args, {"arg1": "value1"})

    def test_instantiate_task(self):
        task = instantiate_task(self.tools, 1, "tool1", "arg1='value1'")
        self.assertEqual(task["idx"], 1)
        self.assertEqual(task["tool"].name, "tool1")
        self.assertEqual(task["args"], {"arg1": "value1"})

    def test_parse(self):
        text = "1. tool1(arg1='value1')\n2. tool2(arg2=2)"
        tasks = self.parser.parse(text)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["tool"].name, "tool1")
        self.assertEqual(tasks[1]["tool"].name, "tool2")


if __name__ == "__main__":
    unittest.main()
