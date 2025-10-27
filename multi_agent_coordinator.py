from typing import Dict, Any, List, Optional

from pydantic import BaseModel
from agents.base_agent import BaseAgent
from graph import MessagesState
from langchain.messages import HumanMessage, AIMessage
from models.task_models import TaskRequest
from models.orchestrator_models import AgentProcessResult
from routing.intent_classifier import get_intent_classifier
from utils.logger import get_logger
from fallback.base import FallbackStrategy, DefaultFallbackStrategy


class MultiAgentCoordinator:
    """Coordinates multiple agents and routes tasks appropriately"""

    def __init__(self, fallback_strategy: Optional[FallbackStrategy] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.intent_classifier = get_intent_classifier()
        self.logger = get_logger(__name__)
        self.fallback_strategy = fallback_strategy or DefaultFallbackStrategy()

    def register_agent(self, agent: BaseAgent):
        """Register a new agent with the coordinator"""
        self.agents[agent.name] = agent
        self.agent_capabilities[agent.name] = agent.capabilities

    def route_task(self, task: Dict[str, Any]) -> Optional[BaseAgent]:
        """
        Route a task to the most appropriate agent.

        Supports both legacy task_type routing and new description-based routing.
        """
        # Try description-based routing first (new way)
        description = task.get("description", "")
        if description:
            agent_name = self.intent_classifier.get_best_agent(description)
            if agent_name and agent_name in self.agents:
                self.logger.info(f"Intent-based routing: '{description}' -> {agent_name}")
                return self.agents[agent_name]

        # Fallback to legacy task_type routing
        for agent in self.agents.values():
            if agent.can_handle(task):
                self.logger.info(f"Legacy routing: task_type '{task.get('task_type')}' -> {agent.name}")
                return agent

        self.logger.warning(f"No agent found for task: {task}")
        return None


    def get_available_agents(self) -> Dict[str, List[str]]:
        """Get list of available agents and their capabilities"""
        return self.agent_capabilities
    
    
    def process_task_request(self, task_request: TaskRequest, agent_name: str = None, planned_tools: List[str] = None, time_budget_s: float = None) -> Dict[str, Any]:

        """
        Process a TaskRequest through the appropriate agent

        Args:
            task_request: The TaskRequest containing the task to be processed
            agent_name: Optional specific agent to use
            planned_tools: Optional list of tools that should be used
        """
        # Create initial state
        state = MessagesState(
            messages=[HumanMessage(content=task_request.task.description)],
            llm_calls=0
        )

        # Route to appropriate agent
        if agent_name:
            agent = self.agents.get(agent_name)
            if not agent:
                return AgentProcessResult(
                    success=False,
                    error=f"Agent {agent_name} not found",
                    agent_used=agent_name
                )
        else:
            # NEW: Use intent-based routing from task description
            task_dict = {
                "description": task_request.task.description,
                "task_type": getattr(task_request.task, 'task_type', None)
            }
            agent = self.route_task(task_dict)
            if not agent:
                return AgentProcessResult(
                    success=False,
                    error=f"No agent available to handle this task. Description: '{task_request.task.description}'",
                    agent_used=None
                )

        # Process with agent
        try:
            result_state = agent.process_task_request(
                state,
                task_request,
                execution_mode=task_request.execution_mode,
                planned_tools=planned_tools,
                time_budget_s=time_budget_s
            )
            result = AgentProcessResult(
                success=True,
                messages=[msg.content for msg in result_state["messages"]],
                agent_used=agent.name,
                llm_calls=result_state.get("llm_calls", 0),
                execution_mode=task_request.execution_mode
            )
        except Exception as e:
            result = AgentProcessResult(
                success=False,
                error=str(e),
                agent_used=agent.name
            )

        if not result.success:
            return AgentProcessResult(**self.fallback_strategy.handle_failure(result.dict(), task_request.task.description))

        return result

    def health_check(self) -> Dict[str, Any]:
        """Health check for all agents"""
        status = {}
        for name, agent in self.agents.items():
            try:
                # Simple health check - could be expanded
                status[name] = {"status": "healthy", "capabilities": agent.capabilities}
            except Exception as e:
                status[name] = {"status": "unhealthy", "error": str(e)}
        return status
    
