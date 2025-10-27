from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
from graph import MessagesState
from langchain.messages import AIMessage
from utils.logger import get_logger
from tools import inject_llm_tools
from utils.tool_runner import run_tool_in_subprocess


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""

    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities

    @abstractmethod
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Determine if this agent can handle the given task"""
        pass

    def process_task_request(self, state: MessagesState, task_request, execution_mode: str = "chain", planned_tools: Optional[List[str]] = None, time_budget_s: Optional[float] = None) -> MessagesState:
        """Process a TaskRequest and return updated state"""
        model_with_tools, tools_by_name = inject_llm_tools(self.get_model(), self.get_tools(), agent_name=self.name)
        logger = get_logger(__name__)

        if not planned_tools:
            raise ValueError(f"Agent {self.name} requires planned_tools from the orchestrator")

        current_messages = state["messages"][:]
        llm_calls = state.get("llm_calls", 0)

        # Set deadline if a time budget is provided
        deadline_ts: Optional[float] = None
        if time_budget_s is not None:
            try:
                # Guard against negative values
                budget = max(0.0, float(time_budget_s))
                deadline_ts = time.time() + budget
                logger.debug(f"Agent {self.name}: time budget set to {budget:.2f}s")
            except Exception:
                deadline_ts = None

        LONG_RUNNING_TOOLS = {
            # tool_name: (module_path, attr_name)
            "video_to_transcript": ("agents.transcription_agent", "video_to_transcript"),
        }

        # Convenience accessor for file path if present
        task_file_path = getattr(getattr(task_request, "task", object()), "file_path", None)

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

            # Provide file_path to tools that can leverage it (cooperative)
            if tool_name == "video_to_transcript" and task_file_path and "video_path" not in tool_args:
                tool_args["video_path"] = task_file_path

            # Provide message content to chat tool if missing
            if tool_name == "chat_normally" and "message" not in tool_args:
                try:
                    tool_args["message"] = task_request.task.get_task_description()
                except Exception:
                    tool_args["message"] = ""

            # Provide user_request to reclarify tool if missing
            if tool_name == "reclarify_prompt" and "user_request" not in tool_args:
                try:
                    tool_args["user_request"] = task_request.task.get_task_description()
                except Exception:
                    tool_args["user_request"] = ""

            # Check remaining time before invoking each tool
            if deadline_ts is not None:
                remaining = deadline_ts - time.time()
                if remaining <= 0:
                    logger.info(f"Agent {self.name}: time budget exhausted before running '{tool_name}', skipping remaining tools")
                    current_messages.append(
                        AIMessage(content=f"Time budget exhausted for agent '{self.name}'. Skipping remaining tools.")
                    )
                    break

            try:
                start = time.time()
                # Use hard-timeout subprocess wrapper for long-running tools if we have a deadline
                if deadline_ts is not None and tool_name in LONG_RUNNING_TOOLS:
                    remaining = max(0.0, deadline_ts - time.time())
                    module_path, attr_name = LONG_RUNNING_TOOLS[tool_name]
                    result = run_tool_in_subprocess(module_path, attr_name, tool_args, timeout_s=remaining)
                else:
                    result = tool.invoke(tool_args)
                duration = time.time() - start
                logger.info(f"Agent {self.name}: executed {tool_name} successfully in {duration:.2f}s")
                logger.info(f"Agent {self.name}: executed {tool_name} successfully")
                current_messages.append(
                    AIMessage(content=f"{tool_name} result: {result}")
                )
            except TimeoutError as err:
                logger.error(f"Agent {self.name}: timeout executing {tool_name}: {err}")
                current_messages.append(
                    AIMessage(content=f"{tool_name} timed out: {err}")
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
