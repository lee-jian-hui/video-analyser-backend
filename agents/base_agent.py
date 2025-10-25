from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
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

    def process(self, state: MessagesState, execution_mode: str = "single", file_path: str = "") -> MessagesState:
        """Process the task and return updated state (legacy method)"""
        # Get agent-specific tools and bind them to the model
        from tools import inject_llm_tools
        model_with_tools, tools_by_name = inject_llm_tools(self.get_model(), self.get_tools())

        # Process with the agent's specific implementation
        return self._process_with_tools(state, model_with_tools, tools_by_name, execution_mode, file_path)

    def process_task_request(self, state: MessagesState, task_request, execution_mode: str = "chain", planned_tools: Optional[List[str]] = None) -> MessagesState:
        """Process a TaskRequest and return updated state"""
        # Get agent-specific tools and bind them to the model
        from tools import inject_llm_tools
        model_with_tools, tools_by_name = inject_llm_tools(self.get_model(), self.get_tools())

        # Process with the agent's specific implementation
        return self._process_task_request(state, model_with_tools, tools_by_name, task_request, execution_mode, planned_tools)

    @abstractmethod
    def _process_with_tools(self, state: MessagesState, model_with_tools, tools_by_name, execution_mode: str, file_path: str = "") -> MessagesState:
        """Agent-specific processing logic with tools (legacy method)"""
        pass

    @abstractmethod
    def _process_task_request(self, state: MessagesState, model_with_tools, tools_by_name, task_request, execution_mode: str, planned_tools: Optional[List[str]] = None) -> MessagesState:
        """Agent-specific processing logic for TaskRequest"""
        pass

    @abstractmethod
    def get_model(self):
        """Get the model instance for this agent"""
        pass

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Return list of tools this agent can use"""
        pass