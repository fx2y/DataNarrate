import logging
import os
from typing import Dict, Any, Optional

from langchain_core.tools import BaseTool

from context_manager import ContextManager
from execution_engine import ExecutionEngine, StepResult
from intent_classifier import IntentClassifier
from output_generator import OutputGenerator
from reasoning_engine import ReasoningEngine
from task_planner import TaskPlanner, TaskPlan, TaskStep
from tool_selector import ToolSelector


class PlanAndExecute:
    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None, **kwargs):
        self.logger = logger or logging.getLogger(__name__)
        self.intent_classifier = IntentClassifier(model_name, **kwargs)
        self.task_planner = TaskPlanner(self.intent_classifier, model_name, **kwargs)
        self.context_manager = ContextManager("plan_execute_thread")
        self.tool_selector = ToolSelector(model_name, **kwargs)
        self.execution_engine = ExecutionEngine(logger=self.logger)
        self.reasoning_engine = ReasoningEngine(model_name, **kwargs)
        self.output_generator = OutputGenerator(model_name, **kwargs)

        # Initialize tool registry
        self.initialize_tool_registry()

    def initialize_tool_registry(self):
        # Define and register tools
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

        self.tool_selector.register_tool(SQLQueryTool())
        self.tool_selector.register_tool(VisualizationTool())
        self.tool_selector.register_tool(DataAnalysisTool())

    def plan_step(self, task: str) -> TaskPlan:
        """
        Break down a complex task into a series of steps.
        """
        try:
            self.logger.info(f"Planning steps for task: {task}")
            intent = self.intent_classifier.classify(task)
            context = self.context_manager.get_context_summary()
            available_capabilities = list(set(tool.description for tool in self.tool_selector.tool_registry.values()))

            task_plan = self.task_planner.create_plan(task, intent.intent, available_capabilities, context)
            self.context_manager.update_state(current_task=task)

            self.logger.info(f"Generated plan with {len(task_plan.steps)} steps")
            self.logger.info(f"Planning reasoning: {task_plan.reasoning}")
            return task_plan
        except Exception as e:
            self.logger.error(f"Error in plan_step: {e}", exc_info=True)
            raise

    def execute_step(self, step: TaskStep) -> StepResult:
        """
        Execute a single step of the plan using the specified tool.
        """
        try:
            self.logger.info(f"Executing step {step.step_number}: {step.description}")
            tool_and_input = self.tool_selector.select_tool_for_step(step)
            if not tool_and_input:
                raise ValueError(f"No suitable tool found for step: {step.description}")

            tool, tool_input = tool_and_input
            result = self.execution_engine.execute_step(step, tool, tool_input)
            self.context_manager.update_state(last_tool_used=tool.name)
            self.context_manager.add_relevant_data(f"step_{step.step_number}_result", result.result.output)
            return result
        except Exception as e:
            self.logger.error(f"Error in execute_step: {e}", exc_info=True)
            raise

    def replan_step(self, original_plan: TaskPlan, feedback: str) -> TaskPlan:
        """
        Revise the plan based on feedback and new context.
        """
        try:
            self.logger.info("Replanning based on feedback")
            context = self.context_manager.get_context_summary()
            revised_plan = self.task_planner.replan(original_plan, feedback)
            self.logger.info(f"Generated revised plan with {len(revised_plan.steps)} steps")
            self.logger.info(f"Replanning reasoning: {revised_plan.reasoning}")
            return revised_plan
        except Exception as e:
            self.logger.error(f"Error in replan_step: {e}", exc_info=True)
            raise

    def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """
        Execute the entire plan and generate output.
        """
        results = self.execution_engine.execute_plan(plan.steps, self.tool_selector)

        for step_result in results:
            reasoning_output = self.reasoning_engine.reason(
                self.context_manager.state.current_task,
                {f"step_{step_result.step_number}": step_result.result.output},
                list(self.tool_selector.tool_registry.keys())
            )

            if reasoning_output.confidence < 0.7:
                self.logger.warning(f"Low confidence in step {step_result.step_number}. Replanning...")
                feedback = f"Low confidence in step {step_result.step_number}. Please revise the plan."
                plan = self.replan_step(plan, feedback)
                self.logger.info(f"Revised plan reasoning: {plan.reasoning}")
                # Re-execute the plan from this step
                remaining_steps = [step for step in plan.steps if step.step_number >= step_result.step_number]
                new_results = self.execution_engine.execute_plan(remaining_steps, self.tool_selector)
                results = results[:step_result.step_number - 1] + new_results

        final_output = self.output_generator.generate_output(
            self.context_manager.state.current_task,
            {f"step_{r.step_number}": r.result.output for r in results},
            self.context_manager.state.user_preferences.get("expertise", "general")
        )

        return final_output.dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    planner = PlanAndExecute("deepseek-chat", openai_api_base='https://api.deepseek.com',
                             openai_api_key=os.environ["DEEPSEEK_API_KEY"])

    # Example usage
    task = "Analyze Q2 sales data and create a visualization of top-performing products"

    # Generate and execute plan
    try:
        initial_plan = planner.plan_step(task)
        print("Initial Plan:")
        for step in initial_plan.steps:
            print(f"Step {step.step_number}: {step.description}")
        print()

        # Execute the plan
        final_output = planner.execute_plan(initial_plan)
        print("Final Output:")
        print(final_output)

        # Simulate feedback and replanning
        feedback = "We need to include a comparison with Q1 performance in the analysis."
        revised_plan = planner.replan_step(initial_plan, feedback)
        print("Revised Plan:")
        for step in revised_plan.steps:
            print(f"Step {step.step_number}: {step.description}")
        print()
    except Exception as e:
        print(f"An error occurred: {e}")
