import json
import logging
from typing import Dict, Any, Optional, List

from environs import Env
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from config import config
from context_manager import ContextManager
from execution_engine import ExecutionEngine, StepResult, ToolResult
from intent_classifier import IntentClassifier
from output_generator import OutputGenerator
from query_analyzer import QueryAnalyzer
from reasoning_engine import ReasoningEngine
from schema_retriever import SchemaRetriever
from storyline_creator import StorylineCreator  # New import
from task_planner import TaskPlanner, TaskPlan, TaskStep
from tool_selector import ToolSelector
from visualization_generator import VisualizationGenerator, VisualizationSpec

env = Env()
env.read_env()  # read .env file, if it exists


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, VisualizationSpec):
            return obj.dict()
        elif isinstance(obj, StepResult):
            return {
                "step_number": obj.step_number,
                "result": self.default(obj.result)
            }
        elif isinstance(obj, ToolResult):
            return {
                "output": self.default(obj.output),
                "error": obj.error
            }
        return super().default(obj)


class PlanAndExecute:
    def __init__(self, model_name: str = None, logger: Optional[logging.Logger] = None, **kwargs):
        self.logger = logger or logging.getLogger(__name__)
        model_name = model_name or config.LLM_MODEL_NAME
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.2,
            openai_api_base=config.OPENAI_API_BASE,
            openai_api_key=config.OPENAI_API_KEY,
            **kwargs
        )
        self.intent_classifier = IntentClassifier(self.llm, logger=self.logger)
        self.context_manager = ContextManager(self.intent_classifier, "plan_execute_thread", logger=self.logger)
        self.query_analyzer = QueryAnalyzer(self.llm)
        self.task_planner = TaskPlanner(self.llm, self.context_manager, logger=self.logger)
        self.tool_selector = ToolSelector(self.llm, logger=self.logger)
        self.visualization_generator = VisualizationGenerator(self.llm, logger=self.logger)
        self.execution_engine = ExecutionEngine(self.intent_classifier, logger=self.logger)
        self.reasoning_engine = ReasoningEngine(self.llm, logger=self.logger)
        self.output_generator = OutputGenerator(self.llm, logger=self.logger)
        self.storyline_creator = StorylineCreator(self.llm, logger=self.logger)  # New attribute

        # Initialize tool registry
        self.initialize_tool_registry()

        # Initialize SchemaRetriever
        self.schema_retriever = SchemaRetriever(logger=self.logger)
        self.compressed_schema = None  # Will store the retrieved and compressed schema

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

            def _run(self, data: Dict[str, Any], requirements: str,
                     user_preferences: Dict[str, Any]) -> VisualizationSpec:
                return self.visualization_generator.generate_visualization(data, requirements, user_preferences)

        # class DataAnalysisTool(BaseTool):
        #     name = "Data Analysis Tool"
        #     description = "Performs statistical analysis on datasets"
        #
        #     def _run(self, dataset: str, analysis_type: str) -> str:
        #         return f"Performed {analysis_type} analysis on {dataset}"

        self.tool_selector.register_tool(SQLQueryTool())
        self.tool_selector.register_tool(VisualizationTool())
        # self.tool_selector.register_tool(DataAnalysisTool())

    def retrieve_and_cache_compressed_schema(self):
        if not self.compressed_schema:
            unified_schema = self.schema_retriever.retrieve_unified_schema(
                config.MYSQL_DATABASE,
                config.ELASTICSEARCH_INDEX_PATTERN
            )
            self.compressed_schema = self.schema_retriever.compress_schema(unified_schema)
            self.context_manager.update_schema_info(self.compressed_schema)
        return self.compressed_schema

    def plan_step(self, task: str) -> TaskPlan:
        """
        Break down a complex task into a series of steps.
        """
        try:
            self.logger.info(f"Planning steps for task: {task}")
            intent_classification = self.intent_classifier.classify(task)
            if intent_classification is None:
                raise ValueError("Failed to classify intents")

            intents = intent_classification.intents
            compressed_schema = self.retrieve_and_cache_compressed_schema()
            query_analysis = self.query_analyzer.analyze_query(task, intents, compressed_schema)
            if query_analysis is None:
                raise ValueError("Failed to analyze query")

            task_plan = self.task_planner.plan_task(query_analysis, compressed_schema)
            if task_plan is None:
                raise ValueError("Failed to create task plan")

            self.context_manager.update_state(current_task=task)

            self.logger.info(f"Generated plan with {len(task_plan.steps)} steps")
            self.logger.info(f"Planning reasoning: {task_plan.reasoning}")
            return task_plan
        except Exception as e:
            self.logger.error(f"Error in plan_step: {e}", exc_info=True)
            raise

    def execute_step(self, step: TaskStep, previous_results: List[StepResult]) -> StepResult:
        """
        Execute a single step of the plan using the specified tool, considering previous results for tool selection.
        """
        try:
            self.logger.info(f"Executing step {step.step_number}: {step.description}")
            tool_and_input = self.tool_selector.select_tool_for_step(step, [r.result.output for r in previous_results])
            if not tool_and_input:
                raise ValueError(f"No suitable tool found for step: {step.description}")

            tool, tool_input = tool_and_input
            compressed_schema = self.retrieve_and_cache_compressed_schema()
            result = self.execution_engine.execute_step(step, tool, tool_input, compressed_schema)
            self.context_manager.update_state(last_tool_used=tool.name)
            self.context_manager.add_relevant_data(f"step_{step.step_number}_result", result.result.output)
            return result
        except Exception as e:
            self.logger.error(f"Error in execute_step: {e}", exc_info=True)
            raise

    def replan_step(self, original_plan: TaskPlan, feedback: str, current_step: int) -> TaskPlan:
        """
        Revise the plan based on feedback and new context.
        """
        try:
            self.logger.info("Replanning based on feedback")
            revised_plan = self.task_planner.replan(original_plan, feedback, current_step)
            self.logger.info(f"Generated revised plan with {len(revised_plan.steps)} steps")
            self.logger.info(f"Replanning reasoning: {revised_plan.reasoning}")
            return revised_plan
        except Exception as e:
            self.logger.error(f"Error in replan_step: {e}", exc_info=True)
            raise

    def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """
        Execute the entire plan, perform reasoning after each step, and generate output.
        """
        compressed_schema = self.retrieve_and_cache_compressed_schema()
        results = []
        step_number = 1
        max_replans = 1  # Maximum number of replans allowed
        replan_count = 0

        while step_number <= len(plan.steps):
            if replan_count >= max_replans:
                user_feedback = self.ask_user_for_feedback()
                if user_feedback.lower() == 'abort':
                    self.logger.warning("Execution aborted by user")
                    break
                replan_count = 0  # Reset replan count after user feedback

            step = plan.steps[step_number - 1]
            step_result = self.execute_step(step, results)  # Pass previous results to execute_step
            results.append(step_result)

            reasoning_output = self.reasoning_engine.reason(
                self.context_manager.state.current_task,
                json.loads(
                    json.dumps({f"step_{step_result.step_number}": step_result.result.output}, cls=CustomJSONEncoder)),
                list(self.tool_selector.tool_registry.keys())
            )

            if reasoning_output.confidence < 0.7 or reasoning_output.result_quality < 0.6:
                self.logger.warning(f"Issues detected in step {step_result.step_number}. Replanning...")
                feedback = (
                    f"Step {step_result.step_number} did not produce satisfactory results. "
                    f"Confidence: {reasoning_output.confidence}, Result Quality: {reasoning_output.result_quality}. "
                    f"Reasoning evaluation: {reasoning_output.evaluation}\n"
                    f"Reasoning explanation: {reasoning_output.explanation}\n"
                    f"Please revise the plan considering this reasoning."
                )
                plan = self.replan_step(plan, feedback, current_step=step_number)
                self.logger.info(f"Revised plan reasoning: {plan.reasoning}")
                # Start execution from the beginning of the new plan
                step_number = 1
                results = []
                replan_count += 1
            else:
                step_number += 1

        formatted_results = {f"step_{result.step_number}": result.result.output for result in results}

        final_output = self.output_generator.generate_output(
            self.context_manager.state.current_task,
            json.loads(json.dumps(formatted_results, cls=CustomJSONEncoder)),
            self.context_manager.state.user_preferences.get("expertise", "general")
        )

        # Generate storyline
        context = self.context_manager.state.current_task
        audience = self.context_manager.state.user_preferences.get("audience", "general")
        storyline = self.storyline_creator.create_storyline(formatted_results, context, audience)

        complete_output = {
            "output": final_output,
            "storyline": None
        }

        if storyline and self.storyline_creator.validate_storyline(storyline):
            self.logger.info("Valid storyline generated and added to final output")
            complete_output["storyline"] = storyline.dict()
        else:
            self.logger.warning("Failed to generate a valid storyline")

        if complete_output["storyline"]:
            self.logger.info(f"Storyline title: {complete_output['storyline']['title']}")
            self.logger.info(f"Number of key insights: {len(complete_output['storyline']['key_insights'])}")

            high_relevance_insights = [
                insight for insight in complete_output['storyline']['key_insights']
                if insight['relevance'] > 0.8
            ]
            complete_output["high_relevance_insights"] = high_relevance_insights

        return complete_output

    def ask_user_for_feedback(self) -> str:
        """
        Ask the user for feedback when the maximum number of replans is reached.
        """
        print("The system has reached the maximum number of replanning attempts.")
        print("Please provide feedback or type 'abort' to stop execution:")
        return input().strip()


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    planner = PlanAndExecute()

    # Example usage
    task = "Create a line chart showing the economic growth trends for all provinces from 2018 to 2023."

    # Generate and execute plan
    try:
        initial_plan = planner.plan_step(task)
        print("Initial Plan:")
        for step in initial_plan.steps:
            print(f"Step {step.step_number}: {step.description}")
            print(f"  Required Capability: {step.required_capability}")
            print(f"  Tools: {', '.join(step.tools)}")
        print(f"\nReasoning: {initial_plan.reasoning}")
        print()

        # Execute the plan
        complete_output = planner.execute_plan(initial_plan)
        print("Final Output:")
        print(complete_output["output"])
        if complete_output["storyline"]:
            print("\nStoryline:")
            print(f"Title: {complete_output['storyline']['title']}")
            print(f"Summary: {complete_output['storyline']['summary']}")
            # ... (print other storyline details as needed)

        # Simulate feedback and replanning
        feedback = "Add a stacked bar chart comparing the number of micro and small industries across provinces for the year 2022."
        revised_plan = planner.replan_step(initial_plan, feedback)
        print("\nRevised Plan:")
        for step in revised_plan.steps:
            print(f"Step {step.step_number}: {step.description}")
            print(f"  Required Capability: {step.required_capability}")
            print(f"  Tools: {', '.join(step.tools)}")
        print(f"\nReasoning: {revised_plan.reasoning}")
        print()
    except Exception as e:
        print(f"An error occurred: {e}")
