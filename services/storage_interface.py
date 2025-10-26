"""
Storage Interface - Common protocol for all storage implementations

Allows easy swapping between JSON, SQLite, or other storage backends.
"""
from typing import Protocol, Optional, Dict, Any, List
from abc import ABC, abstractmethod


class ChatHistoryStorageInterface(ABC):
    """
    Abstract interface for chat history storage.

    Implementations:
    - JSONChatHistoryStorage: Simple JSON file-based (default)
    - SQLiteChatHistoryStorage: SQLite database (future)
    """

    @abstractmethod
    def save_history(self, video_id: str, history_data: Dict[str, Any]) -> None:
        """Save chat history for a video"""
        pass

    @abstractmethod
    def load_history(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load chat history for a video"""
        pass

    @abstractmethod
    def delete_history(self, video_id: str) -> bool:
        """Delete chat history for a video"""
        pass

    @abstractmethod
    def list_all_histories(self) -> List[Dict[str, Any]]:
        """List all chat histories (summary info only)"""
        pass

    @abstractmethod
    def save_app_state(self, state_data: Dict[str, Any]) -> None:
        """Save app state (last video, etc.)"""
        pass

    @abstractmethod
    def load_app_state(self) -> Optional[Dict[str, Any]]:
        """Load app state"""
        pass

    @abstractmethod
    def cleanup_old_histories(self, days: int) -> int:
        """Delete histories older than specified days"""
        pass


class FileStorageInterface(ABC):
    """
    Abstract interface for file storage.

    Implementations:
    - FileStorage: Local filesystem storage (current)
    - S3FileStorage: Cloud storage (future)
    """

    @abstractmethod
    def save_uploaded_file(self, file_data: bytes, filename: str) -> tuple[str, str]:
        """Save uploaded file, return (file_id, file_path)"""
        pass

    @abstractmethod
    def import_local_file(self, source_path: str, filename: Optional[str] = None, copy_file: bool = True) -> tuple[str, str]:
        """Import local file, return (file_id, file_path)"""
        pass

    @abstractmethod
    def get_file_path(self, file_id: str) -> str:
        """Get file path by file_id"""
        pass

    @abstractmethod
    def file_exists(self, file_id: str) -> bool:
        """Check if file exists"""
        pass

    @abstractmethod
    def delete_file(self, file_id: str) -> bool:
        """Delete file"""
        pass

    @abstractmethod
    def list_files(self) -> Dict[str, str]:
        """List all files"""
        pass

    @abstractmethod
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get file metadata"""
        pass

    @abstractmethod
    def cleanup_old_files(self, days: int) -> int:
        """Cleanup old files"""
        pass
