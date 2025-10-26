from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from graph import MessagesState
from langchain.messages import AIMessage
from utils.logger import get_logger
from tools import inject_llm_tools


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""

    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities

    @abstractmethod
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Determine if this agent can handle the given task"""
        pass

    def process_task_request(self, state: MessagesState, task_request, execution_mode: str = "chain", planned_tools: Optional[List[str]] = None) -> MessagesState:
        """Process a TaskRequest and return updated state"""
        model_with_tools, tools_by_name = inject_llm_tools(self.get_model(), self.get_tools(), agent_name=self.name)
        logger = get_logger(__name__)

        if not planned_tools:
            raise ValueError(f"Agent {self.name} requires planned_tools from the orchestrator")

        current_messages = state["messages"][:]
        llm_calls = state.get("llm_calls", 0)

        for entry in planned_tools:
            if isinstance(entry, dict):
                tool_name = entry.get("name")
                tool_args = entry.get("args", {})
            else:
                tool_name = entry
                tool_args = {}

            tool = tools_by_name.get(tool_name)
            if not tool:
                logger.warning(f"Tool '{tool_name}' not found for agent {self.name}")
                current_messages.append(
                    AIMessage(content=f"Tool '{tool_name}' is not available.")
                )
                continue

            try:
                result = tool.invoke(tool_args)
                logger.info(f"Agent {self.name}: executed {tool_name} successfully")
                current_messages.append(
                    AIMessage(content=f"{tool_name} result: {result}")
                )
            except Exception as err:
                logger.error(f"Agent {self.name}: error executing {tool_name}: {err}")
                current_messages.append(
                    AIMessage(content=f"Error executing {tool_name}: {err}")
                )

        return {"messages": current_messages, "llm_calls": llm_calls}

    @abstractmethod
    def get_model(self):
        """Get the model instance for this agent"""
        pass

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Return list of tools this agent can use"""
        pass
