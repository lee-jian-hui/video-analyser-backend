"""
Shared storage path helpers for the desktop app.

Keeps platform-specific logic (Documents folder, overrides, etc.)
out of individual services so there is a single source of truth.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

# Environment variables to override defaults (useful for tests)
ENV_STORAGE_ROOT = "VIDEO_ANALYZER_STORAGE_ROOT"


def get_storage_root(custom_root: Optional[str] = None) -> Path:
    """
    Return the base directory for all app data (videos, outputs, metadata).

    Priority:
        1. Explicit `custom_root` parameter (allows tests to isolate files)
        2. `VIDEO_ANALYZER_STORAGE_ROOT` environment variable
        3. OS-appropriate Documents folder
    """
    if custom_root:
        root = Path(custom_root)
    elif os.getenv(ENV_STORAGE_ROOT):
        root = Path(os.getenv(ENV_STORAGE_ROOT, ""))
    else:
        root = _default_documents_root()

    root.mkdir(parents=True, exist_ok=True)
    return root


def get_videos_dir(root: Optional[Path] = None) -> Path:
    """Return (and create) the directory where video files should live."""
    root_path = root or get_storage_root()
    videos_dir = root_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    return videos_dir


def get_outputs_dir(root: Optional[Path] = None) -> Path:
    """Return (and create) the directory for generated outputs."""
    root_path = root or get_storage_root()
    outputs_dir = root_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def get_registry_path(root: Optional[Path] = None) -> Path:
    """Return the file path for the video registry metadata store."""
    root_path = root or get_storage_root()
    return root_path / "video_registry.json"


def _default_documents_root() -> Path:
    """OS-specific default under the user's Documents folder."""
    home = Path.home()
    system = platform.system()

    if system == "Windows":
        return home / "Documents" / "VideoAnalyzer"
    if system == "Darwin":
        return home / "Documents" / "VideoAnalyzer"
    return home / "Documents" / "VideoAnalyzer"
