"""
Chat History Storage - JSON Implementation

Stores chat histories in JSON files with rolling summarization.
Easy to swap with SQLite implementation later.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from services.storage_interface import ChatHistoryStorageInterface
from storage_paths import get_storage_root

logger = logging.getLogger(__name__)


class JSONChatHistoryStorage(ChatHistoryStorageInterface):
    """
    JSON file-based chat history storage.

    Directory structure:
        <storage_root>/
            chat_history/
                {video_id}.json    # Individual chat histories
            app_state.json         # Last video, app state
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize JSON-based chat history storage.

        Args:
            base_dir: Optional custom base directory
        """
        storage_root = Path(get_storage_root(base_dir))
        self.history_dir = storage_root / "chat_history"
        self.history_dir.mkdir(exist_ok=True, parents=True)

        self.app_state_file = storage_root / "app_state.json"

        logger.info(f"JSONChatHistoryStorage initialized: {self.history_dir}")

    def save_history(self, video_id: str, history_data: Dict[str, Any]) -> None:
        """
        Save chat history to JSON file.

        Args:
            video_id: Unique video identifier
            history_data: Chat history data (follows ChatHistory model)
        """
        file_path = self.history_dir / f"{video_id}.json"

        try:
            # Ensure updated_at is set
            if "updated_at" not in history_data:
                history_data["updated_at"] = datetime.now().isoformat()

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, default=str, ensure_ascii=False)

            logger.debug(f"Saved chat history for video: {video_id}")

        except Exception as e:
            logger.error(f"Failed to save chat history for {video_id}: {e}")
            raise

    def load_history(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Load chat history from JSON file.

        Args:
            video_id: Unique video identifier

        Returns:
            Chat history data or None if not found
        """
        file_path = self.history_dir / f"{video_id}.json"

        if not file_path.exists():
            logger.debug(f"No chat history found for video: {video_id}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.debug(f"Loaded chat history for video: {video_id}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chat history {video_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load chat history for {video_id}: {e}")
            raise

    def delete_history(self, video_id: str) -> bool:
        """
        Delete chat history file.

        Args:
            video_id: Unique video identifier

        Returns:
            True if deleted, False if not found
        """
        file_path = self.history_dir / f"{video_id}.json"

        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted chat history: {video_id}")
                return True
            else:
                logger.warning(f"Chat history not found for deletion: {video_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete chat history {video_id}: {e}")
            raise

    def list_all_histories(self) -> List[Dict[str, Any]]:
        """
        List all chat histories (summary info only, not full messages).

        Returns:
            List of dicts with video_id, video_path, message count, timestamps
        """
        histories = []

        try:
            for file_path in self.history_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract summary info only
                    summary = {
                        "video_id": data.get("video_id"),
                        "display_name": data.get("display_name", "Unknown"),
                        "video_path": data.get("video_path"),
                        "total_messages": data.get("total_messages", 0),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "has_summary": bool(data.get("conversation_summary"))
                    }
                    histories.append(summary)

                except Exception as e:
                    logger.warning(f"Failed to read history file {file_path.name}: {e}")
                    continue

            # Sort by most recent first
            histories.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            logger.debug(f"Listed {len(histories)} chat histories")

            return histories

        except Exception as e:
            logger.error(f"Failed to list chat histories: {e}")
            return []

    def save_app_state(self, state_data: Dict[str, Any]) -> None:
        """
        Save app state (last video, settings, etc.).

        Args:
            state_data: App state data
        """
        try:
            # Add timestamp
            state_data["last_updated"] = datetime.now().isoformat()

            with open(self.app_state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.debug("Saved app state")

        except Exception as e:
            logger.error(f"Failed to save app state: {e}")
            raise

    def load_app_state(self) -> Optional[Dict[str, Any]]:
        """
        Load app state.

        Returns:
            App state data or None if not found
        """
        if not self.app_state_file.exists():
            logger.debug("No app state found")
            return None

        try:
            with open(self.app_state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.debug("Loaded app state")
            return data

        except Exception as e:
            logger.error(f"Failed to load app state: {e}")
            return None

    def cleanup_old_histories(self, days: int) -> int:
        """
        Delete chat histories older than specified days.

        Args:
            days: Delete histories older than this many days

        Returns:
            Number of histories deleted
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0

        try:
            for file_path in self.history_dir.glob("*.json"):
                try:
                    # Check file modification time
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                    if mtime < cutoff_time:
                        # Extract video_id from filename
                        video_id = file_path.stem

                        if self.delete_history(video_id):
                            deleted_count += 1

                except Exception as e:
                    logger.warning(f"Failed to cleanup history {file_path.name}: {e}")
                    continue

            logger.info(f"Cleaned up {deleted_count} old chat histories (older than {days} days)")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old histories: {e}")
            return 0


# Default instance - can be swapped with SQLite implementation later
def get_chat_history_storage(storage_type: str = "json") -> ChatHistoryStorageInterface:
    """
    Factory function to get chat history storage implementation.

    Args:
        storage_type: "json" or "sqlite" (future)

    Returns:
        ChatHistoryStorageInterface implementation
    """
    if storage_type == "json":
        return JSONChatHistoryStorage()
    elif storage_type == "sqlite":
        # Future implementation
        # from services.sqlite_chat_history_storage import SQLiteChatHistoryStorage
        # return SQLiteChatHistoryStorage()
        raise NotImplementedError("SQLite storage not yet implemented")
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
