from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain.messages import BaseMessage
from graph import MessagesState


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""

    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities

    @abstractmethod
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Determine if this agent can handle the given task"""
        pass

    def process(self, state: MessagesState) -> MessagesState:
        """Process the task and return updated state"""
        # Get agent-specific tools and bind them to the model
        from tools import inject_llm_tools
        model_with_tools, tools_by_name = inject_llm_tools(self.get_model(), self.get_tools())

        # Process with the agent's specific implementation
        return self._process_with_tools(state, model_with_tools, tools_by_name)

    @abstractmethod
    def _process_with_tools(self, state: MessagesState, model_with_tools, tools_by_name) -> MessagesState:
        """Agent-specific processing logic with tools"""
        pass

    @abstractmethod
    def get_model(self):
        """Get the model instance for this agent"""
        pass

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Return list of tools this agent can use"""
        pass