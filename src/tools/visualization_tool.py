import json
from typing import List, Dict, Any, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState


class VisualizationRequest(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate (e.g., 'bar', 'line', 'scatter')")
    data: Dict[str, List[Any]] = Field(...,
                                       description="Data to visualize, with keys as column names and values as lists of data points")
    title: Optional[str] = Field(None, description="Title of the chart")
    x_axis_label: Optional[str] = Field(None, description="Label for the x-axis")
    y_axis_label: Optional[str] = Field(None, description="Label for the y-axis")


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
             state: Optional[InjectedState[Any]] = None) -> str:
        try:
            if chart_type not in ["bar", "line", "scatter"]:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            spec = self._generate_spec(chart_type, data, title, x_axis_label, y_axis_label)

            # Use the state if provided (e.g., to store the spec for later use)
            if state:
                state.setdefault("visualizations", []).append(spec)

            return json.dumps(spec)
        except Exception as e:
            error_message = f"Error generating visualization specification: {str(e)}"
            if state:
                state.setdefault("errors", []).append(error_message)
            return error_message

    async def _arun(self, chart_type: str, data: Dict[str, List[Any]], title: Optional[str] = None,
                    x_axis_label: Optional[str] = None, y_axis_label: Optional[str] = None,
                    state: Optional[InjectedState[Any]] = None) -> str:
        # In this case, the operation is not I/O bound, so we can just call the sync version
        return self._run(chart_type, data, title, x_axis_label, y_axis_label, state)


def create_visualization_tool() -> VisualizationTool:
    return VisualizationTool()
