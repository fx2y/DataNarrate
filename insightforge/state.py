from typing import Annotated, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    schema: Optional[str]
    query: Optional[str]
    results: Optional[List[dict]]
    preprocessed_data: Optional[dict]
    visualization_data: Optional[dict]
    narration: Optional[str]
