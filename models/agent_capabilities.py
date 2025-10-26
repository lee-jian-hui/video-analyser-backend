"""
Agent Capabilities Model

Defines the structure for agent capabilities and intent keywords
that enable natural language task routing.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class CapabilityCategory(str, Enum):
    """High-level categories for agent capabilities"""
    AUDIO = "audio"
    VISION = "vision"
    TEXT = "text"
    GENERATION = "generation"
    ANALYSIS = "analysis"


class AgentCapability(BaseModel):
    """
    Structured capability definition for an agent.

    Each agent file should define an instance of this at the top
    to clearly document what it can do and how to route to it.
    """

    # Human-readable capability descriptions
    capabilities: List[str] = Field(
        ...,
        description="List of capabilities in plain English (e.g., 'audio transcription', 'object detection')"
    )

    # Keywords for intent matching
    intent_keywords: List[str] = Field(
        ...,
        description="Keywords that indicate this agent should handle the task"
    )

    # Categories this agent belongs to
    categories: List[CapabilityCategory] = Field(
        default_factory=list,
        description="High-level categories this agent belongs to"
    )

    # Example task descriptions
    example_tasks: List[str] = Field(
        default_factory=list,
        description="Example natural language task descriptions this agent can handle"
    )

    # Priority for routing (higher = preferred when multiple agents match)
    routing_priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Priority when multiple agents can handle a task (1=lowest, 10=highest)"
    )

    def matches_description(self, description: str) -> bool:
        """Check if this capability matches a task description"""
        description_lower = description.lower()
        return any(keyword in description_lower for keyword in self.intent_keywords)

    def get_match_score(self, description: str) -> float:
        """Calculate a match score (0.0 to 1.0) for a task description"""
        description_lower = description.lower()

        # Count matching keywords
        keyword_matches = sum(1 for keyword in self.intent_keywords if keyword in description_lower)

        if keyword_matches == 0:
            return 0.0

        # Score based on keyword density and priority
        keyword_density = keyword_matches / len(self.intent_keywords)
        priority_weight = self.routing_priority / 10.0

        # Weighted score: 70% keyword match, 30% priority
        return (0.7 * keyword_density) + (0.3 * priority_weight)


class AgentCapabilityRegistry:
    """
    Registry to collect all agent capabilities for routing decisions.
    Populated automatically when agents are instantiated.
    """

    _capabilities: dict[str, AgentCapability] = {}

    @classmethod
    def register(cls, agent_name: str, capability: AgentCapability):
        """Register an agent's capabilities"""
        cls._capabilities[agent_name] = capability

    @classmethod
    def get_capability(cls, agent_name: str) -> Optional[AgentCapability]:
        """Get capabilities for a specific agent"""
        return cls._capabilities.get(agent_name)

    @classmethod
    def get_all_capabilities(cls) -> dict[str, AgentCapability]:
        """Get all registered capabilities"""
        return cls._capabilities.copy()

    @classmethod
    def find_matching_agents(cls, description: str, threshold: float = 0.3) -> List[tuple[str, float]]:
        """
        Find agents that can handle a task description.

        Returns:
            List of (agent_name, match_score) tuples, sorted by score descending
        """
        matches = []

        for agent_name, capability in cls._capabilities.items():
            score = capability.get_match_score(description)
            if score >= threshold:
                matches.append((agent_name, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches
