from abc import ABC, abstractmethod
from typing import Dict, Any


class FallbackStrategy(ABC):
    """Strategy interface for handling agent processing failures."""

    @abstractmethod
    def handle_failure(self, result: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Transform a failed agent result into a graceful response."""
        raise NotImplementedError


class DefaultFallbackStrategy(FallbackStrategy):
    """Fallback that simply returns the original error result."""

    def handle_failure(self, result: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        return result
