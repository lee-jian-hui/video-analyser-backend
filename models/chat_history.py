"""
Chat History Model

Pure data models for chat history. Business logic lives in services.
"""
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
import logging

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
    Chat history data model.

    Strategy handled by service layer:
    - Keep last N messages in full (recent_messages)
    - Summarize older messages into conversation_summary
    - Auto-summarize as messages exceed limits
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

    # Configuration (persisted with the model)
    MAX_RECENT_MESSAGES: int = Field(default=10)
    SUMMARIZE_THRESHOLD: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True
