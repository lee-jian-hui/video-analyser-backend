"""
Reclarify Agent

Handles ambiguous requests, missing prerequisites (like video file), and
general conversational responses via explicit tool calls to reduce hallucination
and unify control flow.
"""

from typing import Any, Dict, List
from langchain.tools import tool
from langchain.messages import HumanMessage

from agents.base_agent import BaseAgent
from llm import get_llm_model
from utils.logger import get_logger
from models.agent_capabilities import AgentCapability, CapabilityCategory, AgentCapabilityRegistry
from templates.orchestrator_prompts import OrchestratorPrompts


RECLARIFY_AGENT_CAPABILITIES = AgentCapability(
    capabilities=[
        "Clarify ambiguous user requests",
        "General conversation and guidance",
        "Ask for missing inputs (e.g., video)",
    ],
    intent_keywords=[
        "clarify", "clarification", "ask", "question", "help", "explain",
        "what", "how", "why", "chat", "talk", "conversation",
    ],
    categories=[CapabilityCategory.TEXT],
    example_tasks=[
        "I'm not sure what I need",
        "Can you help me decide what to do?",
        "Explain how this app works",
        "I don't have a video yet, what next?",
    ],
    routing_priority=5,
)


logger = get_logger(__name__)


def _capability_summary() -> str:
    from models.agent_capabilities import AgentCapabilityRegistry
    capabilities = AgentCapabilityRegistry.get_all_capabilities()
    if not capabilities:
        return "- No specialized agents are currently registered."
    lines: List[str] = []
    for agent_name, capability in capabilities.items():
        summary = ", ".join(capability.capabilities[:3]) or "No documented capabilities"
        display_name = agent_name.replace("_", " ").title()
        lines.append(f"- {display_name}: {summary}")
    return "\n".join(lines)


@tool
def reclarify_prompt(user_request: str = "") -> str:
    """Generate a concise clarification prompt guiding the user to provide clearer goals or missing inputs."""
    summary = _capability_summary()
    return OrchestratorPrompts.format_clarification_message(user_request or "", summary)


@tool
def missing_video() -> str:
    """Check for a loaded video and ask the user to upload/select one if missing."""
    from context.video_context import get_video_context
    vc = get_video_context()
    path = vc.get_current_video_path()
    if not path:
        return (
            "No video detected. Please upload or register a video file "
            "to continue. Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm"
        )
    return f"A video is loaded: {path}. What would you like to do with it?"


@tool
def chat_normally(message: str) -> str:
    """Have a brief, helpful conversation without executing specialized tools."""
    model = get_llm_model()
    prompt = f"The user says: '{message}'. Provide a concise, helpful reply."
    resp = model.invoke([HumanMessage(content=prompt)])
    return resp.content


class ReclarifyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="reclarify_agent",
            capabilities=RECLARIFY_AGENT_CAPABILITIES.capabilities,
        )
        self.capability_definition = RECLARIFY_AGENT_CAPABILITIES
        self.model = get_llm_model()
        self.tools = [reclarify_prompt, missing_video, chat_normally]

        AgentCapabilityRegistry.register(self.name, self.capability_definition)

    def can_handle(self, task: Dict[str, Any]) -> bool:
        # Reclarify agent can handle generic/ambiguous or chatty requests
        description = task.get("description", "")
        return bool(description)

    def get_model(self):
        return self.model

    def get_tools(self) -> List[Any]:
        return self.tools

