"""
Chat History Model

Manages conversation history with automatic summarization.
Uses rolling summary to keep context window manageable.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import logging

from services.chat_history_storage import get_chat_history_storage, ChatHistoryStorageInterface

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What objects are in this video?",
                "timestamp": "2025-01-01T12:00:00"
            }
        }


class ChatHistory(BaseModel):
    """
    Chat history with rolling summarization.

    Strategy:
    - Keep last N messages in full (recent_messages)
    - Summarize older messages into conversation_summary
    - When messages exceed limit, auto-summarize oldest ones
    """

    # Video identification
    video_id: str
    video_path: str
    display_name: str = ""

    # Rolling summary of old messages
    conversation_summary: str = ""

    # Recent messages (full fidelity)
    recent_messages: List[ChatMessage] = Field(default_factory=list)

    # Metadata
    total_messages: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Configuration
    MAX_RECENT_MESSAGES: int = Field(default=10)
    SUMMARIZE_THRESHOLD: int = Field(default=5)  # Summarize when we have 5+ old messages

    class Config:
        arbitrary_types_allowed = True

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat history.

        Automatically triggers summarization when recent_messages exceeds MAX_RECENT_MESSAGES.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        message = ChatMessage(role=role, content=content)
        self.recent_messages.append(message)
        self.total_messages += 1
        self.updated_at = datetime.now().isoformat()

        # Auto-summarize if we exceed the limit
        if len(self.recent_messages) > self.MAX_RECENT_MESSAGES:
            self._auto_summarize()

        logger.debug(f"Added {role} message to chat history (total: {self.total_messages})")

    def _auto_summarize(self) -> None:
        """
        Automatically summarize oldest messages.

        Takes SUMMARIZE_THRESHOLD oldest messages, creates/updates summary,
        and removes them from recent_messages.
        """
        if len(self.recent_messages) <= self.MAX_RECENT_MESSAGES:
            return

        # How many to summarize
        to_summarize_count = len(self.recent_messages) - self.MAX_RECENT_MESSAGES + self.SUMMARIZE_THRESHOLD
        to_summarize = self.recent_messages[:to_summarize_count]

        logger.info(f"Auto-summarizing {len(to_summarize)} messages")

        try:
            # Create summary of these messages
            new_summary = self._summarize_messages(to_summarize)

            # Merge with existing summary
            if self.conversation_summary:
                self.conversation_summary = self._merge_summaries(
                    self.conversation_summary,
                    new_summary
                )
            else:
                self.conversation_summary = new_summary

            # Remove summarized messages
            self.recent_messages = self.recent_messages[to_summarize_count:]

            logger.info(f"Summary updated. Recent messages: {len(self.recent_messages)}")

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

        prompt = f"""Summarize this conversation excerpt concisely (2-3 sentences):

{messages_text}

Focus on:
1. What the user asked about
2. Key findings or results
3. Any important context

Summary:"""

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

        prompt = f"""Merge these two conversation summaries into one concise summary (3-4 sentences max):

Previous summary:
{old_summary}

New summary:
{new_summary}

Combined summary:"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            # Fallback: simple concatenation
            return f"{old_summary} {new_summary}"

    def get_context_for_llm(self) -> str:
        """
        Get formatted context string for LLM.

        Returns conversation summary + recent messages in a format
        suitable for including in LLM prompts.

        Returns:
            Formatted context string
        """
        parts = []

        if self.conversation_summary:
            parts.append(f"**Previous Conversation Summary:**\n{self.conversation_summary}\n")

        if self.recent_messages:
            parts.append("**Recent Messages:**")
            for msg in self.recent_messages:
                parts.append(f"- {msg.role.upper()}: {msg.content}")

        return "\n".join(parts) if parts else "No previous conversation."

    def save(self, storage: Optional[ChatHistoryStorageInterface] = None) -> None:
        """
        Save chat history to storage.

        Args:
            storage: Optional custom storage instance (uses default if None)
        """
        if storage is None:
            storage = get_chat_history_storage()

        storage.save_history(self.video_id, self.dict())
        logger.debug(f"Saved chat history for {self.video_id}")

    @classmethod
    def load(cls, video_id: str, storage: Optional[ChatHistoryStorageInterface] = None) -> Optional['ChatHistory']:
        """
        Load chat history from storage.

        Args:
            video_id: Video identifier
            storage: Optional custom storage instance (uses default if None)

        Returns:
            ChatHistory instance or None if not found
        """
        if storage is None:
            storage = get_chat_history_storage()

        data = storage.load_history(video_id)
        if data is None:
            return None

        return cls(**data)

    @classmethod
    def create_new(cls, video_id: str, video_path: str, display_name: str = "") -> 'ChatHistory':
        """
        Create a new chat history.

        Args:
            video_id: Unique video identifier
            video_path: Path to video file
            display_name: Human-readable display name

        Returns:
            New ChatHistory instance
        """
        return cls(
            video_id=video_id,
            video_path=video_path,
            display_name=display_name or video_id
        )

    def to_summary_dict(self) -> dict:
        """
        Get summary info (without full messages) for listing.

        Returns:
            Dict with summary information
        """
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "display_name": self.display_name,
            "total_messages": self.total_messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_summary": bool(self.conversation_summary),
            "recent_message_count": len(self.recent_messages)
        }
