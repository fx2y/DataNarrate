import logging
import os
from typing import Any, Dict, Optional, List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from intent_classifier import IntentClassifier
from task_planner import TaskStep
from tool_selector import ToolSelector


class ToolResult(BaseModel):
    output: Any = Field(description="The output of the tool execution")
    error: Optional[str] = Field(default=None, description="Error message if the execution failed")


class StepResult(BaseModel):
    step_number: int = Field(description="The number of the step in the plan")
    result: ToolResult = Field(description="The result of the tool execution for this step")


class ExecutionEngine:
    def __init__(self, intent_classifier: IntentClassifier, logger: Optional[logging.Logger] = None,
                 max_retries: int = 3):
        self.intent_classifier = intent_classifier
        self.logger = logger or logging.getLogger(__name__)
        self.max_retries = max_retries

    def execute_tool(self, tool: BaseTool, **kwargs) -> ToolResult:
        """
        Execute a given tool with the provided arguments.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                self.logger.info(f"Executing tool: {tool.name}")
                result = tool.invoke(kwargs)
                self.logger.info(f"Tool executed successfully: {tool.name}")
                return ToolResult(output=result)
            except Exception as e:
                retries += 1
                self.logger.error(f"Error executing tool {tool.name}: {e}. Retry {retries}/{self.max_retries}")
                if retries >= self.max_retries:
                    self.logger.error(f"Max retries reached for tool {tool.name}. Failing execution.")
                    return ToolResult(error=str(e))

    def execute_step(self, step: TaskStep, tool: BaseTool, tool_input: Dict[str, Any]) -> StepResult:
        """
        Execute a single step from the plan.
        """
        result = self.execute_tool(tool, **tool_input)
        return StepResult(step_number=step.step_number, result=result)

    def execute_plan(self, plan: List[TaskStep], tool_selector: ToolSelector) -> List[StepResult]:
        """
        Execute a plan consisting of multiple tool executions.
        """
        results = []
        for step in plan:
            tool_and_input = tool_selector.select_tool_for_step(step)
            if not tool_and_input:
                self.logger.error(f"No suitable tool found for step {step.step_number}")
                break
            tool, tool_input = tool_and_input
            result = self.execute_step(step, tool, tool_input)
            results.append(result)

            if result.result.error:
                self.logger.warning(f"Step {result.step_number} failed: {step}. Stopping plan execution.")
                break

        return results

    def execute(self, query: str, context: dict):
        intent_classification = self.intent_classifier.classify(query)
        if intent_classification.intent == "data_retrieval":
            return self.execute_data_retrieval(query, context)
        elif intent_classification.intent == "visualization":
            return self.execute_visualization(query, context)
        # ... handle other intents ...

    def execute_data_retrieval(self, query: str, context: dict):
        # Implementation for data retrieval
        pass

    def execute_visualization(self, query: str, context: dict):
        # Implementation for visualization
        pass

    # ... other intent-specific execution methods ...


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    classifier = IntentClassifier("deepseek-chat", openai_api_base='https://api.deepseek.com',
                                  openai_api_key=os.environ["DEEPSEEK_API_KEY"])
    engine = ExecutionEngine(classifier)
    selector = ToolSelector("deepseek-chat", openai_api_base='https://api.deepseek.com',
                            openai_api_key=os.environ["DEEPSEEK_API_KEY"])


    # Example tools
    class SQLQueryTool(BaseTool):
        name = "SQL Query Tool"
        description = "Executes SQL queries on a database"

        def _run(self, query: str) -> str:
            return f"Executed SQL query: {query}"


    class VisualizationTool(BaseTool):
        name = "Visualization Tool"
        description = "Creates data visualizations and charts"

        def _run(self, data: Dict[str, Any]) -> str:
            return f"Created visualization for data: {data}"


    class DataAnalysisTool(BaseTool):
        name = "Data Analysis Tool"
        description = "Performs statistical analysis on datasets"

        def _run(self, dataset: str, analysis_type: str) -> str:
            return f"Performed {analysis_type} analysis on {dataset}"


    selector.register_tool(SQLQueryTool())
    selector.register_tool(VisualizationTool())
    selector.register_tool(DataAnalysisTool())

    plan = [
        TaskStep(step_number=1, description="Query Q2 sales data", required_capability="data_query",
                 input_description={"query": "Sales data for Q2"}),
        TaskStep(step_number=2, description="Analyze top performers", required_capability="data_analysis",
                 input_description={"dataset": "Q2 sales data", "analysis_type": "top performers"}),
        TaskStep(step_number=3, description="Visualize top products", required_capability="data_visualization",
                 input_description={"data": "Top 10 products by revenue"})
    ]

    results = engine.execute_plan(plan, selector)
    for result in results:
        if result.result.error:
            print(f"Step {result.step_number} failed: {result.result.error}")
        else:
            print(f"Step {result.step_number} succeeded: {result.result.output}")
