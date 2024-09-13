from typing import Annotated, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: Optional[str]
    visualization_data: Optional[dict]
    narration: Optional[str]

class SchemaState(TypedDict):
    schema: Optional[str]
    state: State

class ResultsState(TypedDict):
    results: List[dict]
    state: State