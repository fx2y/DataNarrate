import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt import ToolNode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationRequest(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate (e.g., 'bar', 'line', 'scatter')")
    data: Dict[str, List[Any]] = Field(...,
                                       description="Data to visualize, with keys as column names and values as lists of data points")
    title: Optional[str] = Field(None, description="Title of the chart")
    x_axis_label: Optional[str] = Field(None, description="Label for the x-axis")
    y_axis_label: Optional[str] = Field(None, description="Label for the y-axis")


class VisualizationState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage]] = Field(default_factory=list)
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class VisualizationTool(BaseTool):
    name = "visualization_generator"
    description = "Generates visualization specifications based on provided data and requirements"
    args_schema = VisualizationRequest

    def _generate_spec(self, chart_type: str, data: Dict[str, List[Any]], title: Optional[str],
                       x_axis_label: Optional[str], y_axis_label: Optional[str]) -> Dict[str, Any]:
        spec = {
            "chart_type": chart_type,
            "data": data,
            "layout": {}
        }
        if title:
            spec["layout"]["title"] = title
        if x_axis_label:
            spec["layout"]["xaxis"] = {"title": x_axis_label}
        if y_axis_label:
            spec["layout"]["yaxis"] = {"title": y_axis_label}
        return spec

    def _run(self, chart_type: str, data: Dict[str, List[Any]], title: Optional[str] = None,
             x_axis_label: Optional[str] = None, y_axis_label: Optional[str] = None,
             state: Optional[InjectedState("VisualizationState")] = None) -> str:
        try:
            if chart_type not in ["bar", "line", "scatter"]:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            spec = self._generate_spec(chart_type, data, title, x_axis_label, y_axis_label)

            if state:
                state.visualizations.append(spec)
                logger.info(f"Added visualization spec to state. Total specs: {len(state.visualizations)}")

            return json.dumps(spec)
        except Exception as e:
            error_message = f"Error generating visualization specification: {str(e)}"
            if state:
                state.errors.append(error_message)
                logger.error(f"Error in visualization generation: {error_message}")
            return error_message

    async def _arun(self, chart_type: str, data: Dict[str, List[Any]], title: Optional[str] = None,
                    x_axis_label: Optional[str] = None, y_axis_label: Optional[str] = None,
                    state: Optional[InjectedState(VisualizationState)] = None) -> str:
        # Simulate an asynchronous operation
        await asyncio.sleep(1)
        return self._run(chart_type, data, title, x_axis_label, y_axis_label, state)


def create_visualization_tool() -> VisualizationTool:
    return VisualizationTool()


# Graph setup
def create_visualization_graph():
    workflow = StateGraph(VisualizationState)

    # Create tool node
    tool_node = ToolNode(tools=[create_visualization_tool()])
    workflow.add_node("visualization_tool", tool_node)

    # Add edges
    workflow.add_edge("visualization_tool", END)

    # Set entry point
    workflow.set_entry_point("visualization_tool")

    return workflow.compile()


# Example usage
async def main():
    graph = create_visualization_graph()

    initial_state = VisualizationState(
        messages=[
            HumanMessage(content="Create a bar chart of sales data")
        ]
    )

    result = await graph.ainvoke(initial_state)

    print("Visualizations:", result.visualizations)
    print("Errors:", result.errors)


if __name__ == "__main__":
    asyncio.run(main())
