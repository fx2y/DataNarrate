import json
import operator
from typing import Annotated, List, Tuple, TypedDict, Literal, Dict, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from datanarrate.config import config
from datanarrate.schema_retriever import SchemaRetriever
from nodes.analyze_query import analyze_query, QueryAnalysis
from nodes.execute_step import execute_step, ExecuteStepState
from nodes.generate_output import generate_output, OutputState, OutputFormat
from nodes.plan_task import plan_task_and_select_tools, PlanningState, Plan
from nodes.reason import ReasoningNode, ReasoningConfig, ReasoningState
# Assume we have these tool implementations
from tools.mysql_tool import create_mysql_tool, MySQLConfig
from tools.visualization_tool import VisualizationTool


class PlanExecuteState(TypedDict):
    input: str
    query_analysis: Optional[QueryAnalysis]
    plan: Optional[Plan]
    current_step: int
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    messages: Annotated[List[AIMessage | HumanMessage], operator.add]
    intermediate_steps: Annotated[List[Dict], operator.add]
    output: Optional[OutputFormat]
    response: str
    replan_count: int
    tool_outputs: Dict[str, Any]


# Initialize tools and models
mysql_config = MySQLConfig(
    host=config.MYSQL_HOST,
    port=3306,
    user=config.MYSQL_USER,
    password=config.MYSQL_PASSWORD,
    database=config.MYSQL_DATABASE
)
mysql_tools = create_mysql_tool(config=mysql_config)
visualization_tool = VisualizationTool()

tools = [
    mysql_tools,
    visualization_tool,
]
llm = ChatOpenAI(
    model_name=config.LLM_MODEL_NAME,
    openai_api_base=config.OPENAI_API_BASE,
    openai_api_key=config.OPENAI_API_KEY,
    temperature=0.2
)

schema_retriever = SchemaRetriever()
unified_schema = schema_retriever.retrieve_unified_schema(
    config.MYSQL_DATABASE,
    config.ELASTICSEARCH_INDEX_PATTERN
)
schema = schema_retriever.compress_schema(unified_schema)


# Create nodes
async def analyze_node(state: PlanExecuteState, config: RunnableConfig) -> Dict[str, Any]:
    result = await analyze_query(state, config)
    return {"query_analysis": result["query_analysis"]}


async def plan_node(state: PlanExecuteState) -> Dict[str, Any]:
    planning_state = PlanningState(
        messages=state["messages"],
        query_analysis=state["query_analysis"]
    )
    result = await plan_task_and_select_tools(planning_state, llm, tools)
    return {"plan": result.plan, "current_step": 0}


reasoning_node = ReasoningNode(ReasoningConfig(llm=llm))


async def execute_node(state: PlanExecuteState) -> Dict[str, Any]:
    current_step = state["plan"].steps[state["current_step"]]

    # Resolve input dependencies
    for tool in current_step.tools:
        for arg_name, arg_value in tool.args.items():
            if isinstance(arg_value, str) and arg_value.startswith("$"):
                step_index = int(arg_value[1:])
                for output_key, output_value in state["tool_outputs"].items():
                    if output_key.endswith(f"_output_{step_index}"):
                        tool.args[arg_name] = output_value
                        break

    execute_state = ExecuteStepState(
        messages=state["messages"],
        intermediate_steps=state["intermediate_steps"],
        current_step=current_step
    )
    result = await execute_step(execute_state, tools)

    # Store tool outputs for potential future use
    new_tool_outputs = state["tool_outputs"].copy()
    for tool_result in result["intermediate_steps"]:
        tool_name = tool_result["action"]["tool"]
        new_tool_outputs[f"{tool_name}_output_{state['current_step']}"] = tool_result["observation"]

    return {
        "messages": result["messages"],
        "intermediate_steps": result["intermediate_steps"],
        "current_step": state["current_step"] + 1,
        "tool_outputs": new_tool_outputs,
        "past_steps": state["past_steps"] + [(current_step.description, str(result["intermediate_steps"]))]
    }


async def output_node(state: PlanExecuteState) -> Dict[str, Any]:
    output_state = OutputState(
        messages=state["messages"],
        analysis_results=str(state["past_steps"])
    )
    result = await generate_output(output_state)
    return {"output": result["output"], "response": result["output"].summary}


async def reason_node(state: PlanExecuteState) -> Dict[str, Any]:
    reasoning_state = ReasoningState(
        context={
            "current_step": state["current_step"],
            "plan": state["plan"],
            "past_steps": state["past_steps"],
            "tool_outputs": state["tool_outputs"]
        },
        messages=state["messages"]
    )
    result = await reasoning_node(reasoning_state)
    return {
        "messages": state["messages"] + result["messages"],
        "reasoning_output": result["reasoning_output"]
    }


def should_continue(state: PlanExecuteState) -> Literal[
    "reason", "execute_step", "replan", "generate_output", "__end__"]:
    if state["output"]:
        return "__end__"
    elif state["current_step"] < len(state["plan"].steps):
        return "reason"
    elif state["replan_count"] < 3:  # Allow up to 3 replans
        return "replan"
    else:
        return "generate_output"


async def replan_node(state: PlanExecuteState) -> Dict[str, Any]:
    planning_state = PlanningState(
        messages=state["messages"],
        query_analysis=state["query_analysis"]
    )
    result = await plan_task_and_select_tools(planning_state, llm, tools)
    return {
        "plan": result.plan,
        "current_step": 0,
        "replan_count": state["replan_count"] + 1
    }


# Create the graph
workflow = StateGraph(PlanExecuteState)

workflow.add_node("analyze", analyze_node)
workflow.add_node("planning", plan_node)
workflow.add_node("reason", reason_node)
workflow.add_node("execute_step", execute_node)
workflow.add_node("generate_output", output_node)
workflow.add_node("replan", replan_node)

workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "planning")
workflow.add_edge("planning", "execute_step")
workflow.add_edge("replan", "execute_step")

workflow.add_conditional_edges(
    "reason",
    lambda x: x["reasoning_output"].next_action,
    {
        "continue": "execute_step",
        "revise": "replan",
        "finish": "generate_output",
    }
)

workflow.add_conditional_edges(
    "execute_step",
    should_continue,
    {
        "reason": "reason",
        "replan": "replan",
        "generate_output": "generate_output",
        "__end__": END
    }
)

workflow.add_edge("generate_output", END)

app = workflow.compile()


# Usage
async def run_plan_execute_agent(query: str):
    initial_state = PlanExecuteState(
        input=query,
        query_analysis=None,
        plan=None,
        current_step=0,
        past_steps=[],
        messages=[HumanMessage(content=query)],
        intermediate_steps=[],
        output=None,
        response="",
        replan_count=0,
        tool_outputs={}
    )
    result = await app.ainvoke(initial_state, config={"configurable": {"schema_info": schema}})
    return result


# Example usage
async def main():
    result = await run_plan_execute_agent("Analyze the sales data for Q1 2023 and provide insights with visualizations")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
