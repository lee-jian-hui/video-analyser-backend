from langchain.tools import tool
from utils.logger import get_logger


class SafeToolWrapper:
    """Wrap tool invocations with consistent logging/exception handling."""

    def __init__(self, tool_obj, agent_name: str):
        self._tool = tool_obj
        self.name = getattr(tool_obj, "name", getattr(tool_obj, "__name__", "tool"))
        self._agent_name = agent_name
        self._logger = get_logger(__name__)

    def invoke(self, args=None, **kwargs):
        self._logger.info("Agent %s: running tool %s", self._agent_name, self.name)
        try:
            result = self._tool.invoke(args, **kwargs)
            self._logger.info("Agent %s: tool %s completed", self._agent_name, self.name)
            return result
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Agent %s: tool %s failed", self._agent_name, self.name)
            raise


def inject_llm_tools(model, additional_tools=None, agent_name: str = "agent"):
    # Augment the LLM with tools
    base_tools = []

    # Add any additional tools from agents
    if additional_tools:
        base_tools.extend(additional_tools)

    model_with_tools = model.bind_tools(base_tools)

    wrapped_tools = {
        tool.name: SafeToolWrapper(tool, agent_name)
        for tool in base_tools
    }

    return model_with_tools, wrapped_tools

