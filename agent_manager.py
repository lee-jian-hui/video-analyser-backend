"""
Agent Manager - Hot loading and health monitoring for AI agents
"""

import logging
from typing import Dict, Optional, Any, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import threading
import time

from ai_model_manager import AIModelManager, AIModelType
from configs import Config


class AgentStatus(Enum):
    """Agent status states"""
    UNLOADED = "unloaded"      # Not loaded yet
    LOADING = "loading"        # Currently loading
    READY = "ready"           # Loaded and ready
    ERROR = "error"           # Failed to load
    DISABLED = "disabled"     # Intentionally disabled


@dataclass
class AgentInfo:
    """Information about an agent"""
    name: str
    status: AgentStatus
    model_dependencies: List[AIModelType]
    last_used: Optional[datetime] = None
    error_message: Optional[str] = None
    instance: Optional[Any] = None


class AgentManager:
    """Manages hot loading and health monitoring of AI agents"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_manager = AIModelManager()
        self._agents: Dict[str, AgentInfo] = {}
        self._lock = threading.Lock()

        # Register available agents
        self._register_agents()

    def _register_agents(self):
        """Register all available agents with their dependencies"""
        self._agents = {
            "vision_agent": AgentInfo(
                name="vision_agent",
                status=AgentStatus.UNLOADED,
                model_dependencies=[AIModelType.OBJECT_DETECTION, AIModelType.LLM]
            ),
            "transcription_agent": AgentInfo(
                name="transcription_agent",
                status=AgentStatus.UNLOADED,
                model_dependencies=[AIModelType.TRANSCRIPTION, AIModelType.LLM]
            )
        }

    def startup_health_check(self) -> Dict[str, bool]:
        """
        Run health checks for all models at startup.
        Returns dict of model_type -> healthy status
        """
        self.logger.info("Running startup health checks...")
        health_status = {}

        # Check each model type
        for model_type in AIModelType:
            try:
                if model_type == AIModelType.TRANSCRIPTION:
                    model = self.model_manager.get_whisper_model()
                    health_status[model_type.value] = model is not None

                elif model_type == AIModelType.OBJECT_DETECTION:
                    model = self.model_manager.get_yolo_model()
                    health_status[model_type.value] = model is not None

                elif model_type == AIModelType.LLM:
                    if Config.USE_LOCAL_LLM:
                        if Config.LOCAL_MODEL_TYPE.lower() == "codellama":
                            model = self.model_manager.get_codellama_model()
                        else:
                            model = self.model_manager.get_llama_model()
                        health_status[model_type.value] = model is not None
                    else:
                        # Gemini API - assume healthy if API key present
                        health_status[model_type.value] = bool(Config.GEMINI_API_KEY)

            except Exception as e:
                self.logger.error(f"Health check failed for {model_type.value}: {e}")
                health_status[model_type.value] = False

        # Update agent statuses based on health checks
        self._update_agent_health(health_status)

        return health_status

    def _update_agent_health(self, health_status: Dict[str, bool]):
        """Update agent availability based on model health"""
        with self._lock:
            for agent_name, agent_info in self._agents.items():
                # Check if all required models are healthy
                all_deps_healthy = all(
                    health_status.get(dep.value, False)
                    for dep in agent_info.model_dependencies
                )

                if all_deps_healthy:
                    if agent_info.status == AgentStatus.UNLOADED:
                        self.logger.info(f"Agent {agent_name} dependencies ready")
                else:
                    agent_info.status = AgentStatus.DISABLED
                    missing_deps = [
                        dep.value for dep in agent_info.model_dependencies
                        if not health_status.get(dep.value, False)
                    ]
                    agent_info.error_message = f"Missing dependencies: {missing_deps}"
                    self.logger.warning(f"Agent {agent_name} disabled - {agent_info.error_message}")

    def get_agent(self, agent_name: str):
        """
        Hot load and return an agent instance.
        Only loads the agent when first requested.
        """
        with self._lock:
            if agent_name not in self._agents:
                raise ValueError(f"Unknown agent: {agent_name}")

            agent_info = self._agents[agent_name]

            # Return cached instance if available
            if agent_info.status == AgentStatus.READY and agent_info.instance:
                agent_info.last_used = datetime.now()
                return agent_info.instance

            # Don't load if disabled
            if agent_info.status == AgentStatus.DISABLED:
                raise RuntimeError(f"Agent {agent_name} is disabled: {agent_info.error_message}")

            # Load the agent
            return self._load_agent(agent_name)

    def _load_agent(self, agent_name: str):
        """Load an agent instance"""
        agent_info = self._agents[agent_name]
        agent_info.status = AgentStatus.LOADING

        try:
            self.logger.info(f"Hot loading agent: {agent_name}")

            if agent_name == "vision_agent":
                from agents.vision_agent import VisionAgent
                instance = VisionAgent()

            elif agent_name == "transcription_agent":
                from agents.transcription_agent import TranscriptionAgent
                instance = TranscriptionAgent()

            else:
                raise ValueError(f"Unknown agent type: {agent_name}")

            # Success
            agent_info.instance = instance
            agent_info.status = AgentStatus.READY
            agent_info.last_used = datetime.now()
            agent_info.error_message = None

            self.logger.info(f"Successfully loaded agent: {agent_name}")
            return instance

        except Exception as e:
            agent_info.status = AgentStatus.ERROR
            agent_info.error_message = str(e)
            agent_info.instance = None
            self.logger.error(f"Failed to load agent {agent_name}: {e}")
            raise

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        with self._lock:
            status = {}
            for agent_name, agent_info in self._agents.items():
                status[agent_name] = {
                    "status": agent_info.status.value,
                    "dependencies": [dep.value for dep in agent_info.model_dependencies],
                    "last_used": agent_info.last_used.isoformat() if agent_info.last_used else None,
                    "error": agent_info.error_message,
                    "loaded": agent_info.instance is not None
                }
            return status

    def unload_agent(self, agent_name: str):
        """Unload an agent to free memory"""
        with self._lock:
            if agent_name in self._agents:
                agent_info = self._agents[agent_name]
                if agent_info.instance:
                    self.logger.info(f"Unloading agent: {agent_name}")
                    agent_info.instance = None
                    agent_info.status = AgentStatus.UNLOADED

    def get_available_agents(self) -> List[str]:
        """Get list of agents that can be loaded"""
        with self._lock:
            return [
                name for name, info in self._agents.items()
                if info.status not in [AgentStatus.DISABLED, AgentStatus.ERROR]
            ]


# Global instance
_agent_manager = None

def get_agent_manager() -> AgentManager:
    """Get the global agent manager instance"""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager()
    return _agent_manager