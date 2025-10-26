"""
Video Registrar Service

Handles registering local video files selected from the desktop app,
stores them via FileStorage, and persists lightweight metadata.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.file_storage import FileStorage
from services.video_registry_store import (
    JSONFileVideoRegistry,
    VideoRegistryStore,
)
from storage_paths import get_registry_path

logger = logging.getLogger(__name__)


class VideoRegistrar:
    """Coordinates video ingestion and metadata persistence."""

    def __init__(
        self,
        file_storage: Optional[FileStorage] = None,
        registry_store: Optional[VideoRegistryStore] = None,
    ) -> None:
        self.file_storage = file_storage or FileStorage()
        default_store = JSONFileVideoRegistry(get_registry_path())
        self.registry_store = registry_store or default_store
        self._records: Dict[str, Dict[str, Any]] = self.registry_store.load_records()

    def register_local_file(
        self,
        source_path: str,
        display_name: Optional[str] = None,
        copy_file: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a local file selected by the frontend.

        Args:
            source_path: Absolute path chosen by the user.
            display_name: Friendly name to display in the UI.
            copy_file: Whether to copy the file into the managed storage directory.
            extra_metadata: Optional structured metadata supplied by callers.
        """
        file_id, stored_path = self.file_storage.import_local_file(
            source_path=source_path,
            filename=display_name,
            copy_file=copy_file,
        )

        path_obj = Path(source_path).expanduser().resolve()
        stored_obj = Path(stored_path)
        metadata = {
            "file_id": file_id,
            "display_name": display_name or path_obj.name,
            "source_path": str(path_obj),
            "stored_path": str(stored_obj),
            "copied": copy_file,
            "size_bytes": stored_obj.stat().st_size,
            "registered_at": time.time(),
        }

        if extra_metadata:
            metadata["extra"] = extra_metadata

        self._records[file_id] = metadata
        self.registry_store.save_records(self._records)
        logger.info("Registered video %s (%s)", metadata["display_name"], file_id)
        return metadata

    def get_video(self, file_id: str) -> Dict[str, Any]:
        """Return metadata for a single file."""
        if file_id not in self._records:
            raise KeyError(f"File {file_id} not found in registry")
        return self._records[file_id]

    def list_videos(self) -> List[Dict[str, Any]]:
        """Return all registered videos sorted by registration time (newest first)."""
        return sorted(
            self._records.values(),
            key=lambda item: item.get("registered_at", 0),
            reverse=True,
        )

    def remove_video(self, file_id: str, delete_file: bool = True) -> None:
        """Remove metadata and optionally delete the stored file."""
        metadata = self._records.pop(file_id, None)
        if not metadata:
            raise KeyError(f"File {file_id} not found in registry")

        copied = metadata.get("copied", True)
        if delete_file and copied:
            self.file_storage.delete_file(file_id)
        else:
            self.file_storage.forget_file(file_id)

        self.registry_store.save_records(self._records)
        logger.info("Removed video %s (deleted=%s)", file_id, delete_file and copied)
