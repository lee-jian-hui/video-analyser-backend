from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VideoChunk(_message.Message):
    __slots__ = ("data", "filename", "chunk_index")
    DATA_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    filename: str
    chunk_index: int
    def __init__(self, data: _Optional[bytes] = ..., filename: _Optional[str] = ..., chunk_index: _Optional[int] = ...) -> None: ...

class UploadResponse(_message.Message):
    __slots__ = ("file_id", "success", "message")
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    file_id: str
    success: bool
    message: str
    def __init__(self, file_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ...) -> None: ...

class RegisterVideoRequest(_message.Message):
    __slots__ = ("file_path", "display_name", "reference_only")
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_ONLY_FIELD_NUMBER: _ClassVar[int]
    file_path: str
    display_name: str
    reference_only: bool
    def __init__(self, file_path: _Optional[str] = ..., display_name: _Optional[str] = ..., reference_only: bool = ...) -> None: ...

class RegisterVideoResponse(_message.Message):
    __slots__ = ("file_id", "stored_path", "display_name", "copied", "size_bytes", "registered_at", "message")
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    STORED_PATH_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    COPIED_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_AT_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    file_id: str
    stored_path: str
    display_name: str
    copied: bool
    size_bytes: int
    registered_at: float
    message: str
    def __init__(self, file_id: _Optional[str] = ..., stored_path: _Optional[str] = ..., display_name: _Optional[str] = ..., copied: bool = ..., size_bytes: _Optional[int] = ..., registered_at: _Optional[float] = ..., message: _Optional[str] = ...) -> None: ...

class ChatRequest(_message.Message):
    __slots__ = ("message", "file_id", "context")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    message: str
    file_id: str
    context: str
    def __init__(self, message: _Optional[str] = ..., file_id: _Optional[str] = ..., context: _Optional[str] = ...) -> None: ...

class ChatResponse(_message.Message):
    __slots__ = ("type", "content", "agent_name", "result_json")
    class ResponseType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        MESSAGE: _ClassVar[ChatResponse.ResponseType]
        PROGRESS: _ClassVar[ChatResponse.ResponseType]
        RESULT: _ClassVar[ChatResponse.ResponseType]
        ERROR: _ClassVar[ChatResponse.ResponseType]
    MESSAGE: ChatResponse.ResponseType
    PROGRESS: ChatResponse.ResponseType
    RESULT: ChatResponse.ResponseType
    ERROR: ChatResponse.ResponseType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    RESULT_JSON_FIELD_NUMBER: _ClassVar[int]
    type: ChatResponse.ResponseType
    content: str
    agent_name: str
    result_json: str
    def __init__(self, type: _Optional[_Union[ChatResponse.ResponseType, str]] = ..., content: _Optional[str] = ..., agent_name: _Optional[str] = ..., result_json: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LastSessionResponse(_message.Message):
    __slots__ = ("has_session", "video_id", "video_name", "video_path", "message_count", "last_updated")
    HAS_SESSION_FIELD_NUMBER: _ClassVar[int]
    VIDEO_ID_FIELD_NUMBER: _ClassVar[int]
    VIDEO_NAME_FIELD_NUMBER: _ClassVar[int]
    VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_FIELD_NUMBER: _ClassVar[int]
    has_session: bool
    video_id: str
    video_name: str
    video_path: str
    message_count: int
    last_updated: str
    def __init__(self, has_session: bool = ..., video_id: _Optional[str] = ..., video_name: _Optional[str] = ..., video_path: _Optional[str] = ..., message_count: _Optional[int] = ..., last_updated: _Optional[str] = ...) -> None: ...

class GetHistoryRequest(_message.Message):
    __slots__ = ("video_id", "include_full_messages")
    VIDEO_ID_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_FULL_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    video_id: str
    include_full_messages: bool
    def __init__(self, video_id: _Optional[str] = ..., include_full_messages: bool = ...) -> None: ...

class GetChatHistoryResponse(_message.Message):
    __slots__ = ("video_id", "video_name", "conversation_summary", "recent_messages", "total_messages", "created_at", "updated_at")
    VIDEO_ID_FIELD_NUMBER: _ClassVar[int]
    VIDEO_NAME_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    RECENT_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    video_id: str
    video_name: str
    conversation_summary: str
    recent_messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    total_messages: int
    created_at: str
    updated_at: str
    def __init__(self, video_id: _Optional[str] = ..., video_name: _Optional[str] = ..., conversation_summary: _Optional[str] = ..., recent_messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., total_messages: _Optional[int] = ..., created_at: _Optional[str] = ..., updated_at: _Optional[str] = ...) -> None: ...

class ClearHistoryRequest(_message.Message):
    __slots__ = ("video_id",)
    VIDEO_ID_FIELD_NUMBER: _ClassVar[int]
    video_id: str
    def __init__(self, video_id: _Optional[str] = ...) -> None: ...

class ClearHistoryResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("role", "content", "timestamp")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    timestamp: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ..., timestamp: _Optional[str] = ...) -> None: ...
