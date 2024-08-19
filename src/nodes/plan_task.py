# nodes/plan_task.py

from typing import Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# Import the Plan model from a shared models file
from models import Plan

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """For the given objective, come up with a simple step by step plan. 
This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
Do not add any superfluous steps. The result of the final step should be the final answer. 
Make sure that each step has all the information needed - do not skip steps."""),
    ("human", "{input}")
])


def create_planner(llm: BaseChatModel):
    """
    Create a planner chain using the provided language model.

    Args:
        llm (BaseChatModel): The language model to use for planning.

    Returns:
        Runnable: A runnable chain for planning.
    """
    return (
            RunnablePassthrough.assign(
                plan=PLANNER_PROMPT | llm.with_structured_output(Plan)
            )
            | RunnableLambda(lambda x: {
        "plan": x["plan"].steps,
        "current_step": 0,
        "status": "planning_complete"
    })
    )


def plan_task(state: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    """
    Generate a plan based on the input in the state.

    Args:
        state (Dict[str, Any]): The current state of the graph execution.
        llm (BaseChatModel): The language model to use for planning.

    Returns:
        Dict[str, Any]: Updated state with the generated plan.
    """
    planner = create_planner(llm)
    try:
        result = planner.invoke({"input": state.get("input", "")})
        return {**state, **result}
    except Exception as e:
        return {
            **state,
            "error": f"Planning failed: {str(e)}",
            "status": "planning_failed"
        }

# Example usage in graph construction (to be placed in a separate file):
# from nodes.plan_task import plan_task
# from langchain_openai import ChatOpenAI
#
# llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
# graph.add_node("plan_task", lambda state: plan_task(state, llm))
