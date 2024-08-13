import logging
from typing import Dict, Any, Optional, List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from config import config
from datanarrate.query_analyzer import QueryAnalysis
from intent_classifier import IntentClassifier


class ConversationState(BaseModel):
    current_task: str = Field(default="", description="The current task being worked on")
    task_progress: float = Field(default=0.0, description="Progress of the current task (0-1)")
    last_tool_used: str = Field(default="", description="The last tool used in the conversation")
    relevant_data: Dict[str, Any] = Field(default_factory=dict, description="Any relevant data for the current context")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences for the conversation")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="History of the conversation")
    current_intents: Optional[List[str]] = Field(default_factory=list, description="The current intents of the user")
    intent_confidences: Optional[Dict[str, float]] = Field(default_factory=dict,
                                                           description="The confidence of each current intent")
    query_analysis: Optional[QueryAnalysis] = Field(default=None, description="The current query analysis")


class ContextManager:
    def __init__(self, intent_classifier: IntentClassifier, thread_id: str, checkpoint_ns: str = "",
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.thread_id = thread_id
        self.checkpoint_ns = checkpoint_ns
        self.state = ConversationState()
        self.memory_saver = MemorySaver()
        self.config = self._create_config()
        self.intent_classifier = intent_classifier

    def _create_config(self) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "checkpoint_ns": self.checkpoint_ns,
            }
        }

    def update_state(self, **kwargs):
        """
        Update the conversation state with new information.
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
                self.logger.info(f"Updated {key} in conversation state")
            else:
                self.logger.warning(f"Attempted to update non-existent state attribute: {key}")
        self._save_checkpoint()

    def get_state(self) -> ConversationState:
        """
        Retrieve the current conversation state.
        """
        return self.state

    def reset_state(self):
        """
        Reset the conversation state to its initial values.
        """
        self.state = ConversationState()
        self.logger.info("Reset conversation state")
        self._save_checkpoint()

    def add_relevant_data(self, key: str, value: Any):
        """
        Add or update relevant data in the conversation state.
        """
        self.state.relevant_data[key] = value
        self.logger.info(f"Added relevant data: {key}")
        self._save_checkpoint()

    def remove_relevant_data(self, key: str):
        """
        Remove a piece of relevant data from the conversation state.
        """
        if key in self.state.relevant_data:
            del self.state.relevant_data[key]
            self.logger.info(f"Removed relevant data: {key}")
            self._save_checkpoint()
        else:
            self.logger.warning(f"Attempted to remove non-existent relevant data: {key}")

    def update_user_preferences(self, preferences: Dict[str, Any]):
        """
        Update user preferences in the conversation state.
        """
        self.state.user_preferences.update(preferences)
        self.logger.info("Updated user preferences")
        self._save_checkpoint()

    def add_to_conversation_history(self, role: str, content: str):
        """
        Add a new message to the conversation history.
        """
        self.state.conversation_history.append({"role": role, "content": content})
        self.logger.info(f"Added {role} message to conversation history")
        self._save_checkpoint()

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the current context for use in prompts or decision-making.
        """
        return {
            "current_task": self.state.current_task,
            "task_progress": f"{self.state.task_progress * 100:.0f}%",
            "last_tool_used": self.state.last_tool_used,
            "relevant_data": self.state.relevant_data,
            "user_preferences": self.state.user_preferences,
            "conversation_history": [
                {"role": msg["role"],
                 "content": msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]}
                for msg in self.state.conversation_history[-5:]  # Include last 5 messages
            ],
            "current_intents": self.state.current_intents,
            "intent_confidences": self.state.intent_confidences,
        }

    def _save_checkpoint(self):
        """
        Save the current state as a checkpoint.
        """
        checkpoint = {"id": f"{self.thread_id}-{self.checkpoint_ns}-{len(self.state.conversation_history)}"}
        metadata = {}
        new_versions = {}
        self.memory_saver.put(self.config, checkpoint, metadata, new_versions)
        self.logger.debug("Saved checkpoint of current state")

    def load_checkpoint(self):
        """
        Load the last saved checkpoint.
        """
        checkpoint_tuple = self.memory_saver.get_tuple(self.config)
        if checkpoint_tuple:
            self.state = ConversationState(**checkpoint_tuple.checkpoint)
            self.logger.info("Loaded checkpoint")
        else:
            self.logger.warning("No checkpoint found, using initial state")

    def update_context(self, query: str):
        """
        Update the context based on the current query.
        """
        intent_classification = self.intent_classifier.classify(query)
        if intent_classification:
            self.state.current_intents = intent_classification.intents
            self.state.intent_confidences = dict(zip(intent_classification.intents, intent_classification.confidences))
        else:
            self.logger.warning("Failed to classify intents for the query")
            self.state.current_intents = []
            self.state.intent_confidences = {}
        self.add_to_conversation_history("user", query)
        self._save_checkpoint()

    def update_schema_info(self, compressed_schema: Dict[str, Any]):
        """
        Update the schema information in the context.
        """
        self.state.relevant_data["schema_info"] = compressed_schema
        self.logger.info("Updated schema information in context")
        self._save_checkpoint()


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL_NAME,
        openai_api_base=config.OPENAI_API_BASE,
        openai_api_key=config.OPENAI_API_KEY
    )
    classifier = IntentClassifier(llm)
    context_manager = ContextManager(classifier, thread_id="example_thread")

    # Example usage
    context_manager.update_state(current_task="Analyze Q2 sales data", task_progress=0.3,
                                 last_tool_used="SQL Query Tool")
    context_manager.add_relevant_data("q2_sales", {"total": 1000000, "top_product": "Widget A"})
    context_manager.update_user_preferences({"chart_type": "bar", "color_scheme": "blue"})
    context_manager.add_to_conversation_history("user", "Show me the Q2 sales data")
    context_manager.add_to_conversation_history("assistant", "Certainly! I'll analyze the Q2 sales data for you.")

    print(context_manager.get_context_summary())

    # Simulate saving and loading a checkpoint
    context_manager.reset_state()
    print("\nAfter reset:")
    print(context_manager.get_context_summary())

    context_manager.load_checkpoint()
    print("\nAfter loading checkpoint:")
    print(context_manager.get_context_summary())
