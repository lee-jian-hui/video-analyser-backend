from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from graph import MessagesState
from langchain.messages import HumanMessage, AIMessage


class MultiAgentCoordinator:
    """Coordinates multiple agents and routes tasks appropriately"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}

    def register_agent(self, agent: BaseAgent):
        """Register a new agent with the coordinator"""
        self.agents[agent.name] = agent
        self.agent_capabilities[agent.name] = agent.capabilities

    def route_task(self, task: Dict[str, Any]) -> Optional[BaseAgent]:
        """Route a task to the most appropriate agent"""
        for agent in self.agents.values():
            if agent.can_handle(task):
                return agent
        return None

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request from the frontend through appropriate agents"""
        # Create initial state
        state = MessagesState(
            messages=[HumanMessage(content=request.get("content", ""))],
            llm_calls=0
        )

        # Route to appropriate agent
        agent = self.route_task(request)
        if not agent:
            return {
                "success": False,
                "error": "No agent available to handle this task",
                "agent_used": None
            }

        # Process with agent
        try:
            result_state = agent.process(state)
            return {
                "success": True,
                "messages": [msg.content for msg in result_state["messages"]],
                "agent_used": agent.name,
                "llm_calls": result_state.get("llm_calls", 0)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_used": agent.name
            }

    def get_available_agents(self) -> Dict[str, List[str]]:
        """Get list of available agents and their capabilities"""
        return self.agent_capabilities

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