import asyncio
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .config import settings
from .graph import create_data_narration_graph
from .state import DataNarrationState
from .tools import get_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataNarrationSystem:
    def __init__(self):
        self.graph = create_data_narration_graph()
        self.tools = get_tools()
        self.memory = MemorySaver()
        self.llm = ChatAnthropic(model=settings.LLM_MODEL)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def process_query(self, query: str):
        initial_state = DataNarrationState(messages=[HumanMessage(content=query)])
        try:
            async for event in self.graph.astream(initial_state):
                if event.event == "start":
                    logger.info("Starting query processing...")
                elif event.event == "end":
                    logger.info("Query processing complete.")
                    return event.state.output
                else:
                    logger.info(f"Executing {event.name}...")
                    # You can add more detailed logging or progress updates here
        except Exception as e:
            logger.error(f"An error occurred during processing: {str(e)}")
            return f"An error occurred during processing: {str(e)}"

    def run(self, query: str):
        return asyncio.run(self.process_query(query))

    @classmethod
    def from_config(cls, config_path: str):
        # Load configuration from file
        # Implement configuration loading logic here
        return cls()
