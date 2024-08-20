import operator
from typing import Annotated, List, Tuple, Literal, Dict, Any, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from datanarrate.config import config
from datanarrate.schema_retriever import SchemaRetriever
from nodes.analyze_query import analyze_query, QueryAnalysis
from nodes.execute_step import ExecuteStepState
from nodes.generate_output import generate_output, OutputState, OutputFormat
from nodes.plan_task import plan_task_and_select_tools, PlanningState, Plan
from nodes.reason import ReasoningNode, ReasoningConfig, ReasoningState
from src.nodes.execute_step import StepExecutor
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
    reasoning_input: Optional[Dict[str, Any]]
    error_occurred: bool


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

from langchain_core.tools import tool
import pandas as pd
import matplotlib.pyplot as plt


@tool
def data_extractor(source: str, query: str) -> Dict[str, Any]:
    """
    Extracts data from a specified source using a query.

    Args:
    source (str): The data source (e.g., 'mysql', 'postgresql', 'csv', 'api')
    query (str): The query to extract data (SQL query for databases, endpoint for API, etc.)
    parameters (dict, optional): Additional parameters for the query

    Returns:
    Dict[str, Any]: Extracted data as a dictionary representation of a DataFrame
    """
    if source.lower() == 'mysql':
        result = mysql_tools.run(query)
        df = pd.DataFrame(result)
    else:
        # Fallback for other sources or placeholder for future implementations
        df = pd.DataFrame({"column1": [1, 2, 3], "column2": ["a", "b", "c"]})

    return df.to_dict(orient="split")


@tool
def data_transformer(data_inputs: List[Dict[str, Any]], operations: List[Dict]) -> Dict[str, Any]:
    """
    Applies a series of transformation operations to one or more input DataFrames.

    Args:
    data_inputs (List[Dict[str, Any]]): List of input DataFrames as dictionaries
    operations (List[Dict]): List of operations to apply, each as a dictionary
                             e.g., [{'type': 'merge', 'on': 'id'},
                                    {'type': 'filter', 'column': 'age', 'condition': '> 18'},
                                    {'type': 'group_by', 'columns': ['category'], 'agg_func': 'sum'}]

    Returns:
    Dict[str, Any]: Transformed DataFrame as a dictionary
    """
    # Convert dictionary representations back to DataFrames
    dfs = [pd.DataFrame.from_dict(data, orient="split") for data in data_inputs]

    # Apply operations
    result_df = dfs[0]  # Start with the first DataFrame
    for operation in operations:
        if operation['type'] == 'merge':
            result_df = result_df.merge(dfs[1], on=operation['on'])
        elif operation['type'] == 'filter':
            result_df = result_df[result_df[operation['column']].apply(lambda x: eval(f"x {operation['condition']}"))]
        elif operation['type'] == 'group_by':
            result_df = result_df.groupby(operation['columns']).agg(operation['agg_func']).reset_index()
        # Add more operation types as needed

    # Convert the result back to a dictionary representation
    return result_df.to_dict(orient="split")


@tool
def visualizer(data_inputs: List[Dict[str, Any]], chart_type: str, x: str, y: Union[str, List[str]], **kwargs) -> str:
    """
    Creates a visualization based on one or more input DataFrames and specifications.

    Args:
    data_inputs (List[Dict[str, Any]]): List of input DataFrames as dictionaries
    chart_type (str): Type of chart (e.g., 'line', 'bar', 'scatter')
    x (str): Column name for x-axis
    y (Union[str, List[str]]): Column name(s) for y-axis (can be multiple for comparison)
    **kwargs: Additional arguments for customization (e.g., title, color, etc.)

    Returns:
    str: A description or representation of the created visualization
    """
    # Convert dictionary representations back to DataFrames
    dfs = [pd.DataFrame.from_dict(data, orient="split") for data in data_inputs]

    # Create the visualization
    fig, ax = plt.subplots()

    if chart_type == 'line':
        for df in dfs:
            ax.plot(df[x], df[y])
    elif chart_type == 'bar':
        for i, df in enumerate(dfs):
            ax.bar(df[x], df[y], label=f'Dataset {i + 1}')
    elif chart_type == 'scatter':
        for df in dfs:
            ax.scatter(df[x], df[y])

    ax.set_xlabel(x)
    ax.set_ylabel(y if isinstance(y, str) else ', '.join(y))
    ax.set_title(kwargs.get('title', 'Visualization'))

    if len(dfs) > 1:
        ax.legend()

    # Instead of returning the figure object, we'll save it and return a description
    fig.savefig('visualization.png')
    plt.close(fig)

    return "Visualization created and saved as 'visualization.png'"


# @tool
def statistical_analyzer(data_inputs: List[pd.DataFrame], analysis_type: str, columns: List[str] = None) -> Dict:
    """
    Performs statistical analysis on one or more input DataFrames.

    Args:
    data_inputs (List[pd.DataFrame]): List of input DataFrames
    analysis_type (str): Type of analysis (e.g., 'descriptive', 'correlation', 't_test', 'anova')
    columns (List[str], optional): Specific columns to analyze

    Returns:
    Dict: Results of the statistical analysis
    """
    # Implementation details...
    pass


# @tool
def ml_model(train_data: pd.DataFrame, test_data: pd.DataFrame, target: str, features: List[str], model_type: str,
             **kwargs) -> Any:
    """
    Trains and returns a machine learning model, allowing separate train and test datasets.

    Args:
    train_data (pd.DataFrame): Training data
    test_data (pd.DataFrame): Test data
    target (str): Name of the target column
    features (List[str]): List of feature column names
    model_type (str): Type of model (e.g., 'regression', 'classification')
    **kwargs: Additional arguments for model configuration

    Returns:
    Any: Trained machine learning model object and evaluation metrics
    """
    # Implementation details...
    pass


# @tool
def report_generator(data_inputs: List[Union[str, plt.Figure, pd.DataFrame]], format: str = 'markdown') -> str:
    """
    Generates a report combining various types of inputs.

    Args:
    data_inputs (List[Union[str, plt.Figure, pd.DataFrame]]): List of inputs (text insights, visualizations, data tables)
    format (str): Output format (e.g., 'markdown', 'html', 'pdf')

    Returns:
    str: Generated report in the specified format
    """
    # Implementation details...
    pass


tools = [data_extractor, data_transformer, visualizer]

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

executor = StepExecutor(tools)


async def execute_node(state: PlanExecuteState) -> Dict[str, Any]:
    current_step = state["plan"].steps[state["current_step"]]
    execute_state = ExecuteStepState(
        messages=state["messages"],
        intermediate_steps=state["intermediate_steps"],
        tool_outputs=state["tool_outputs"],
        current_step=current_step,
        current_step_index=state["current_step"]
    )
    result = await executor.execute_step(execute_state)

    new_state = {
        "messages": state["messages"] + result["messages"],
        "intermediate_steps": state["intermediate_steps"] + result["intermediate_steps"],
        "tool_outputs": {**state["tool_outputs"], **result["tool_outputs"]},
        "current_step": state["current_step"] + 1,
        "past_steps": state["past_steps"] + [(current_step.description, str(result["intermediate_steps"]))],
        "reasoning_input": None,
        "error_occurred": result.get("error_occurred", False)
    }

    if result["requires_reasoning"]:
        new_state["reasoning_input"] = {
            "context": result["reasoning_context"],
            "messages": state["messages"]
        }

    return new_state


async def output_node(state: PlanExecuteState) -> Dict[str, Any]:
    output_state = OutputState(
        messages=state["messages"],
        analysis_results=str(state["past_steps"])
    )
    result = await generate_output(output_state)
    return {"output": result["output"], "response": result["output"].summary}


async def reason_node(state: PlanExecuteState) -> Dict[str, Any]:
    if not state["reasoning_input"]:
        return state

    reasoning_state = ReasoningState(**state["reasoning_input"])
    result = await reasoning_node(reasoning_state)

    return {
        "messages": state["messages"] + result["messages"],
        "reasoning_output": result["reasoning_output"],
        "reasoning_input": None
    }


def should_continue(state: PlanExecuteState) -> Literal[
    "reason", "execute_step", "replan", "generate_output", "__end__"]:
    if state["error_occurred"]:
        return "generate_output"
    if state["reasoning_input"]:
        return "reason"
    elif state["current_step"] < len(state["plan"].steps):
        if state["current_step"] >= config.MAX_STEPS:  # Add a MAX_STEPS constant to your config
            return "generate_output"
        return "execute_step"
    elif state["replan_count"] < config.MAX_REPLANS:  # Add a MAX_REPLANS constant to your config
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
        "execute_step": "execute_step",
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
        tool_outputs={},
        reasoning_input=None,
        error_occurred=False
    )
    result = await app.ainvoke(initial_state, config={"configurable": {"schema_info": schema}})
    return result


# Example usage
async def main():
    result = await run_plan_execute_agent("Analyze the sales data for Q1 2023 and provide insights with visualizations")
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
