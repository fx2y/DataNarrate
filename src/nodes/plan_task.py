import json
import re
from typing import List, Dict, Any, Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class ToolSelection(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    description: str
    tool: ToolSelection


class Plan(BaseModel):
    steps: List[PlanStep]
    final_answer: str


# Define state
class PlanningState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    plan: Plan = None
    current_step: int = 0
    status: str = "planning"
    query_analysis: Dict[str, Any] = Field(default_factory=dict)


# Prompts
SYSTEM_PROMPT = """For the given objective, create a step-by-step plan and select appropriate tools for each step.
Available tools:
{tool_descriptions}

Query Analysis:
{query_analysis}

Guidelines:
1. Each step should be concise and actionable.
2. Select one tool for each step. If no tool is required, use the 'LLM' tool for reasoning.
3. For tool arguments, you can use:
   - Placeholders like $0, $1, $2, etc., to refer to outputs from specific previous steps.
   - Direct values for primitives, dicts, or lists.
4. When using the data_extractor tool, refer to the required_data_sources from the query analysis to construct accurate queries.
5. Use the data_transformer tool to process and prepare data for visualization or analysis.
6. Use the visualizer tool to create charts and graphs based on the processed data.
7. The final step should provide the answer to the objective using the 'LLM' tool.

Output Format:
Step 1: [Description] | [Tool(arg1=$0, arg2="literal value", ...)]
Step 2: [Description] | [Tool(arg1=$1, arg2={{"key": "value"}}, ...)]
...
Final Answer: [Description of the final answer] | [LLM(input=$n)]

Always follow Output Format no matter what. Do not put any markdown header.
"""

HUMAN_PROMPT = "Objective: {input}"

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT)
])


def get_tool_descriptions(tools: List[BaseTool]) -> str:
    return "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])


def parse_llm_output(output: str) -> Plan:
    lines = output.strip().split('\n')
    steps = []
    final_answer = ""

    def parse_args(args_string: str) -> Dict[str, Any]:
        # Replace $n placeholders with {{$n}} for easier parsing
        args_string = re.sub(r'\$(\d+)', r'{{$\1}}', args_string)

        # Parse the arguments string as JSON
        try:
            args = json.loads(f"{{{args_string}}}")
        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to a simple key-value parsing
            args = {}
            for pair in args_string.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    args[key.strip()] = value.strip()

        # Process the parsed arguments
        for key, value in args.items():
            if isinstance(value, str):
                # Replace {{$n}} with integers
                if value.startswith('{{$') and value.endswith('}}'):
                    args[key] = int(value[3:-2])
            elif isinstance(value, list):
                args[key] = [int(v[3:-2]) if isinstance(v, str) and v.startswith('{{$') and v.endswith('}}') else v for
                             v in value]

        return args

    for line in lines:
        if line.startswith("Step") or line.startswith("| Step"):
            parts = line.split("|", 1)
            description = parts[0].split(":", 1)[1].strip()
            tool_string = parts[1].strip() if len(parts) > 1 else ""

            tool_parts = tool_string.split("(", 1)
            tool_name = tool_parts[0].strip()
            args = {}
            if len(tool_parts) > 1:
                args_string = tool_parts[1].rstrip(")")
                args = parse_args(args_string)

            steps.append(PlanStep(description=description, tool=ToolSelection(tool=tool_name, args=args)))
        elif line.startswith("Final Answer:") or line.startswith("| Final Answer:"):
            final_answer = line.split(":", 1)[1].strip()
            if "|" in final_answer:
                final_answer, tool_string = final_answer.split("|")
                final_answer = final_answer.strip()
                tool_parts = tool_string.strip().split("(", 1)
                tool_name = tool_parts[0].strip()
                args = {}
                if len(tool_parts) > 1:
                    args_string = tool_parts[1].rstrip(")")
                    args = parse_args(args_string)
                steps.append(PlanStep(description="Final Answer", tool=ToolSelection(tool=tool_name, args=args)))

    return Plan(steps=steps, final_answer=final_answer)


async def plan_task_and_select_tools(
        state: PlanningState,
        llm: BaseChatModel,
        tools: List[BaseTool],
) -> PlanningState:
    """
    Generate a plan and select tools based on the input in the state.
    """
    try:
        tool_descriptions = get_tool_descriptions(tools)
        input_message = state.messages[-1].content if state.messages else ""
        prompt_args = {
            "input": input_message,
            "tool_descriptions": tool_descriptions,
            "query_analysis": state.query_analysis
        }
        llm_output = await llm.ainvoke([planner_prompt.format_messages(**prompt_args)[0]])

        plan = parse_llm_output(llm_output.content)

        return PlanningState(
            messages=state.messages + [AIMessage(content=llm_output.content)],
            plan=plan,
            current_step=0,
            status="planning_complete",
            query_analysis=state.query_analysis
        )
    except Exception as e:
        return PlanningState(
            messages=state.messages + [AIMessage(content=f"Planning and tool selection failed: {str(e)}")],
            status="planning_failed",
            query_analysis=state.query_analysis
        )


# Graph construction
def create_planning_graph(llm: BaseChatModel, tools: List[BaseTool]) -> StateGraph:
    workflow = StateGraph(PlanningState)

    workflow.add_node("plan_and_select", lambda state: plan_task_and_select_tools(state, llm, tools))

    workflow.set_entry_point("plan_and_select")

    return workflow
