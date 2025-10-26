"""
Video Context Manager

Manages video file information and application state for tool execution.
This allows tools to be context-aware without requiring explicit file paths.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import threading


class VideoContext:
    """Thread-safe singleton context manager for video processing"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._current_video_path: Optional[str] = None
            self._video_metadata: Dict[str, Any] = {}
            self._session_data: Dict[str, Any] = {}
            self._initialized = True
    
    def set_current_video(self, video_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Set the current video file being processed"""
        self._current_video_path = str(Path(video_path).resolve())
        self._video_metadata = metadata or {}
        
    def get_current_video_path(self) -> Optional[str]:
        """Get the current video file path"""
        return self._current_video_path
    
    def get_video_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current video"""
        return self._video_metadata.copy()
    
    def set_session_data(self, key: str, value: Any):
        """Store session-specific data"""
        self._session_data[key] = value
    
    def get_session_data(self, key: str, default: Any = None) -> Any:
        """Retrieve session-specific data"""
        return self._session_data.get(key, default)
    
    def clear_session(self):
        """Clear all session data"""
        self._current_video_path = None
        self._video_metadata = {}
        self._session_data = {}
    
    def is_video_loaded(self) -> bool:
        """Check if a video is currently loaded"""
        return self._current_video_path is not None and Path(self._current_video_path).exists()


def get_video_context() -> VideoContext:
    """Get the global video context instance"""
    return VideoContext()