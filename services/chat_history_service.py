"""
Chat History Service

Contains business logic for managing chat history.
Separated from the Pydantic models to keep models as pure data structures.
"""
from typing import List, Optional
from datetime import datetime
import logging

from models.chat_history import ChatHistory, ChatMessage
from services.chat_history_storage import get_chat_history_storage, ChatHistoryStorageInterface

logger = logging.getLogger(__name__)


class ChatHistoryService:
    """Service class for managing chat history operations."""

    def __init__(self, storage: Optional[ChatHistoryStorageInterface] = None):
        """
        Initialize the service.

        Args:
            storage: Optional custom storage instance (uses default if None)
        """
        self.storage = storage or get_chat_history_storage()

    def add_message(self, chat_history: ChatHistory, role: str, content: str) -> None:
        """
        Add a message to the chat history.

        Automatically triggers summarization when recent_messages exceeds MAX_RECENT_MESSAGES.

        Args:
            chat_history: ChatHistory instance to update
            role: "user" or "assistant"
            content: Message content
        """
        message = ChatMessage(role=role, content=content)
        chat_history.recent_messages.append(message)
        chat_history.total_messages += 1
        chat_history.updated_at = datetime.now().isoformat()

        # Auto-summarize if we exceed the limit
        if len(chat_history.recent_messages) > chat_history.MAX_RECENT_MESSAGES:
            self._auto_summarize(chat_history)

        logger.debug(f"Added {role} message to chat history (total: {chat_history.total_messages})")

    def _auto_summarize(self, chat_history: ChatHistory) -> None:
        """
        Automatically summarize oldest messages.

        Takes SUMMARIZE_THRESHOLD oldest messages, creates/updates summary,
        and removes them from recent_messages.

        Args:
            chat_history: ChatHistory instance to update
        """
        if len(chat_history.recent_messages) <= chat_history.MAX_RECENT_MESSAGES:
            return

        # How many to summarize
        to_summarize_count = (
            len(chat_history.recent_messages)
            - chat_history.MAX_RECENT_MESSAGES
            + chat_history.SUMMARIZE_THRESHOLD
        )
        to_summarize = chat_history.recent_messages[:to_summarize_count]

        logger.info(f"Auto-summarizing {len(to_summarize)} messages")

        try:
            # Create summary of these messages
            new_summary = self._summarize_messages(to_summarize)

            # Merge with existing summary
            if chat_history.conversation_summary:
                chat_history.conversation_summary = self._merge_summaries(
                    chat_history.conversation_summary,
                    new_summary
                )
            else:
                chat_history.conversation_summary = new_summary

            # Remove summarized messages
            chat_history.recent_messages = chat_history.recent_messages[to_summarize_count:]

            logger.info(f"Summary updated. Recent messages: {len(chat_history.recent_messages)}")

        except Exception as e:
            logger.error(f"Failed to auto-summarize: {e}")
            # Don't fail the add_message operation, just log the error

    def _summarize_messages(self, messages: List[ChatMessage]) -> str:
        """
        Create a summary of given messages using LLM.

        Args:
            messages: List of messages to summarize

        Returns:
            Concise summary string
        """
        # Import here to avoid circular dependency
        from llm import get_chat_llm
        from langchain.messages import HumanMessage

        llm = get_chat_llm()

        # Format messages
        messages_text = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in messages
        ])

        from templates.chat_history_prompts import ChatHistoryPrompts
        prompt = ChatHistoryPrompts.format_summarize_messages_prompt(messages_text)

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            # Fallback: simple concatenation
            return f"User asked about video analysis. Discussed: {', '.join([m.content[:30] for m in messages[:3]])}..."

    def _merge_summaries(self, old_summary: str, new_summary: str) -> str:
        """
        Merge existing summary with new summary.

        Args:
            old_summary: Previous conversation summary
            new_summary: Summary of new messages

        Returns:
            Combined summary
        """
        from llm import get_chat_llm
        from langchain.messages import HumanMessage

        llm = get_chat_llm()

        from templates.chat_history_prompts import ChatHistoryPrompts
        prompt = ChatHistoryPrompts.format_merge_summaries_prompt(old_summary, new_summary)

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            # Fallback: simple concatenation
            return f"{old_summary} {new_summary}"

    def get_context_for_llm(self, chat_history: ChatHistory) -> str:
        """
        Get formatted context string for LLM.

        Returns conversation summary + recent messages in a format
        suitable for including in LLM prompts.

        Args:
            chat_history: ChatHistory instance

        Returns:
            Formatted context string
        """
        parts = []

        if chat_history.conversation_summary:
            parts.append(f"**Previous Conversation Summary:**\n{chat_history.conversation_summary}\n")

        if chat_history.recent_messages:
            parts.append("**Recent Messages:**")
            for msg in chat_history.recent_messages:
                parts.append(f"- {msg.role.upper()}: {msg.content}")

        return "\n".join(parts) if parts else "No previous conversation."

    def save(self, chat_history: ChatHistory, storage: Optional[ChatHistoryStorageInterface] = None) -> None:
        """
        Save chat history to storage.

        Args:
            chat_history: ChatHistory instance to save
            storage: Optional custom storage instance (uses instance storage if None)
        """
        # Enforce configured max saved messages by pruning oldest
        try:
            from configs import Config
            max_saved = max(0, int(getattr(Config, 'CHAT_HISTORY_MAX_SAVED_MESSAGES', 5)))
        except Exception:
            max_saved = 5

        if max_saved >= 0 and len(chat_history.recent_messages) > max_saved:
            # Keep only the last N recent messages
            pruned_count = len(chat_history.recent_messages) - max_saved
            chat_history.recent_messages = chat_history.recent_messages[-max_saved:]
            logger.info(f"Pruned {pruned_count} old messages; saving last {max_saved}")
            # Align total_messages to saved count for accurate UI reporting
            chat_history.total_messages = len(chat_history.recent_messages)

        storage_to_use = storage or self.storage
        storage_to_use.save_history(chat_history.video_id, chat_history.dict())
        logger.debug(f"Saved chat history for {chat_history.video_id}")

    def generate_summary(self, chat_history: ChatHistory, persist: bool = False, storage: Optional[ChatHistoryStorageInterface] = None) -> str:
        """
        Generate a concise summary of the current recent messages.

        Args:
            chat_history: History to summarize
            persist: If True, store into conversation_summary and save
            storage: Optional storage to use when persisting

        Returns:
            Summary text
        """
        if not chat_history.recent_messages:
            return ""

        summary = self._summarize_messages(chat_history.recent_messages)

        if persist and summary:
            if chat_history.conversation_summary:
                chat_history.conversation_summary = self._merge_summaries(
                    chat_history.conversation_summary,
                    summary,
                )
            else:
                chat_history.conversation_summary = summary
            self.save(chat_history, storage=storage)

        return summary

    def load(self, video_id: str, storage: Optional[ChatHistoryStorageInterface] = None) -> Optional[ChatHistory]:
        """
        Load chat history from storage.

        Args:
            video_id: Video identifier
            storage: Optional custom storage instance (uses instance storage if None)

        Returns:
            ChatHistory instance or None if not found
        """
        storage_to_use = storage or self.storage
        data = storage_to_use.load_history(video_id)
        if data is None:
            return None

        history = ChatHistory(**data)
        return history

    def create_new(self, video_id: str, video_path: str, display_name: str = "") -> ChatHistory:
        """
        Create a new chat history.

        Args:
            video_id: Unique video identifier
            video_path: Path to video file
            display_name: Human-readable display name

        Returns:
            New ChatHistory instance
        """
        return ChatHistory(
            video_id=video_id,
            video_path=video_path,
            display_name=display_name or video_id
        )
