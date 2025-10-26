"""
Intent Classifier

Routes tasks to appropriate agents based on natural language descriptions.
Uses the AgentCapabilityRegistry to find matching agents.
"""

from typing import List, Tuple, Optional, Dict, Any
from models.agent_capabilities import AgentCapabilityRegistry
from utils.logger import get_logger


class IntentClassifier:
    """Classifies user intent from task descriptions and routes to appropriate agents"""

    def __init__(self):
        self.logger = get_logger(__name__)

    def classify(self, description: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Classify a task description and find matching agents.

        Args:
            description: Natural language task description
            threshold: Minimum match score (0.0 to 1.0) to consider an agent

        Returns:
            List of (agent_name, match_score) tuples, sorted by score descending
        """
        if not description:
            self.logger.warning("Empty task description provided")
            return []

        # Use the registry to find matching agents
        matches = AgentCapabilityRegistry.find_matching_agents(description, threshold)

        self.logger.debug(f"Intent classification for '{description}': {matches}")

        return matches

    def get_best_agent(self, description: str, threshold: float = 0.3) -> Optional[str]:
        """
        Get the single best agent for a task description.

        Args:
            description: Natural language task description
            threshold: Minimum match score to consider

        Returns:
            Agent name or None if no agent matches
        """
        matches = self.classify(description, threshold)

        if not matches:
            self.logger.warning(f"No agent found for description: '{description}'")
            return None

        best_agent, score = matches[0]
        self.logger.info(f"Best agent for '{description}': {best_agent} (score: {score:.2f})")

        return best_agent

    def get_multiple_agents(
        self,
        description: str,
        threshold: float = 0.3,
        max_agents: int = 3
    ) -> List[str]:
        """
        Get multiple agents that can handle a task (for multi-agent workflows).

        Args:
            description: Natural language task description
            threshold: Minimum match score to consider
            max_agents: Maximum number of agents to return

        Returns:
            List of agent names, sorted by match score
        """
        matches = self.classify(description, threshold)

        # Return up to max_agents
        agent_names = [agent_name for agent_name, _ in matches[:max_agents]]

        self.logger.info(f"Multiple agents for '{description}': {agent_names}")

        return agent_names

    def explain_routing(self, description: str) -> Dict[str, Any]:
        """
        Explain why certain agents were selected for a task.

        Args:
            description: Natural language task description

        Returns:
            Dictionary with routing explanation
        """
        matches = self.classify(description, threshold=0.0)  # Get all matches

        explanation = {
            "description": description,
            "total_agents_checked": len(AgentCapabilityRegistry.get_all_capabilities()),
            "matches": []
        }

        for agent_name, score in matches:
            capability = AgentCapabilityRegistry.get_capability(agent_name)
            if capability:
                matched_keywords = [
                    kw for kw in capability.intent_keywords
                    if kw in description.lower()
                ]

                explanation["matches"].append({
                    "agent": agent_name,
                    "score": score,
                    "matched_keywords": matched_keywords,
                    "capabilities": capability.capabilities,
                    "priority": capability.routing_priority
                })

        return explanation


# Global instance
_intent_classifier = None


def get_intent_classifier() -> IntentClassifier:
    """Get the global intent classifier instance"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
