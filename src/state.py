from typing import Annotated, List, Dict, Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph.message import add_messages


class DataNarrationState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Dict[str, Any] = Field(default_factory=dict)
    task_plan: List[str] = Field(default_factory=list)
    current_step: int = 0
    output: str = ""
    dialog_state: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


def update_dialog_stack(state: DataNarrationState, new_state: str) -> DataNarrationState:
    state.dialog_state.append(new_state)
    return state
