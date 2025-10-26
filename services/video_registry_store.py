"""
Video registry persistence backends.

Allows swapping metadata storage (JSON file today, database later)
without changing the VideoRegistrar API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Protocol

logger = logging.getLogger(__name__)


class VideoRegistryStore(Protocol):
    """Abstraction for persisting registered video metadata."""

    def load_records(self) -> Dict[str, Dict[str, Any]]:
        ...

    def save_records(self, records: Dict[str, Dict[str, Any]]) -> None:
        ...


class JSONFileVideoRegistry(VideoRegistryStore):
    """Persist metadata to a JSON file under the user's storage directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_records(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning("Failed to load registry %s: %s", self.path, exc)
        return {}

    def save_records(self, records: Dict[str, Dict[str, Any]]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
