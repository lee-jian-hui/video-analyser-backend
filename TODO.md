# TODO: Tauri-gRPC Integration Migration

## Overview

Migrate Python backend from local file-based processing to gRPC service for **Tauri-React desktop app** integration.

**Current State:**
- Store output files in local project directory
- Read from files in local project directory
- Global VideoContext singleton (actually fine for single-user desktop app!)
- No chat history persistence across app restarts
- No gRPC communication layer

**Target State:**
- gRPC communication with Tauri desktop frontend
- User-friendly file storage (OS-appropriate directories)
- Chat interface with history persistence across app restarts
- File upload via gRPC streaming
- Resume conversations when app restarts

---

## Current Issues

### 1. File I/O Problems

**Vision Agent** (`agents/vision_agent.py`):
- Line 84: `video_path = video_context.get_current_video_path()` - hardcoded to current working directory
- Line 90: `cap = cv2.VideoCapture(video_path)` - expects local file path (this is fine)

**Transcription Agent** (`agents/transcription_agent.py`):
- Line 76: `video_path = video_context.get_current_video_path()` - hardcoded to current working directory
- Line 134-140: Saves transcript to local project directory (should use user Documents folder)
- Line 94: `video = mp.VideoFileClip(video_path)` - expects local file path (this is fine)

### 2. State Management Issues
- [ ] No chat history persistence across app restarts
- [ ] No multi-file tracking (user may upload multiple videos in one session)
- [ ] File storage in project directory instead of user-friendly locations

**NOTE:** Global VideoContext singleton is FINE for a single-user desktop app. We just need to:
- Track multiple uploaded files (not just one current video)
- Persist state to disk for app restarts
- Use OS-appropriate user directories

### 3. Missing gRPC Layer
- [ ] No protobuf definitions
- [ ] No streaming support for progress updates (transcription/detection are slow)
- [ ] No file upload handling via gRPC

---

## Required Architectural Changes

### A. File Handling Strategy

**Current:**
```python
# Saves to project directory
video_path = video_context.get_current_video_path()
cap = cv2.VideoCapture(video_path)
```

**Target (Desktop App):**
```python
# 1. Tauri uploads file via gRPC streaming
# 2. Backend saves to user Documents folder (OS-appropriate)
# 3. Track multiple uploaded files with file_id
# 4. Return structured data instead of saving to files

# Windows: C:\Users\{user}\Documents\VideoAnalyzer\
# macOS: ~/Documents/VideoAnalyzer/
# Linux: ~/Documents/VideoAnalyzer/

file_storage = FileStorage()  # Uses OS-appropriate paths
video_path = file_storage.get_file_path(file_id)
```

### B. Application State Management (Desktop App)

**New components needed:**
- `FileStorage` - handles OS-appropriate file paths, tracks uploaded files
- `ChatHistoryStore` - persists conversations to SQLite (across app restarts)
- `AppStateManager` - saves/restores app state when app closes/opens

**NOT needed for desktop app:**
- ❌ SessionManager with concurrent session isolation (only one user!)
- ❌ Redis/complex session storage
- ❌ Multi-user authentication

### C. gRPC Service Layer

**New protobuf services:**
```protobuf
service VideoAnalyzerService {
  rpc UploadVideo(stream VideoChunk) returns (UploadResponse);
  rpc SendChatMessage(ChatRequest) returns (stream ChatResponse);
  rpc GetChatHistory(HistoryRequest) returns (ChatHistoryResponse);
  rpc GetUploadedVideos() returns (VideoListResponse);
}
```

### D. Output Format Changes

**Transcription Agent** - return structured data instead of saving to file:
```python
# Current (line 134-140):
with open(transcript_path, 'w') as f:
    f.write(full_transcript)
return f"Saved to: {transcript_path}"

# Should return structured data:
return {
    "transcript_text": transcript_text,
    "segments": formatted_segments,
    "metadata": {"duration": duration, "language": language}
}
```

---

## Frontend Integration MVP - Critical Path

### 🎯 **PRIORITY: What MUST be done before frontend can integrate?**

These are the MINIMUM requirements to get Tauri frontend talking to Python backend:

#### ✅ **Phase 0: Minimal gRPC Integration (1-2 days)**
**Goal:** Basic "hello world" - Tauri can talk to Python backend

- [ ] Install gRPC dependencies
  ```bash
  pip install grpcio grpcio-tools
  ```
- [ ] Create minimal protobuf definition (`protos/video_analyzer.proto`)
  - [ ] `UploadVideo` RPC (file upload)
  - [ ] `SendChatMessage` RPC (basic chat)
  - [ ] Message types: VideoChunk, ChatRequest, ChatResponse
- [ ] Generate Python code from protobuf
- [ ] Create minimal gRPC server (`grpc_service/server.py`)
- [ ] Test with grpcurl to verify server works
- [ ] **Deliverable:** Tauri team can connect to Python backend via gRPC

#### ✅ **Phase 1: File Upload (2-3 days)**
**Goal:** Tauri can upload a video file to Python backend

- [ ] Create `services/file_storage.py`
  - [ ] `save_uploaded_file(file_data, filename)` → (file_id, file_path)
  - [ ] `get_file_path(file_id)` → file_path
  - [ ] Use OS-appropriate user directories (Documents/VideoAnalyzer/)
  - [ ] Track uploaded files in memory (persist later)
- [ ] Implement `UploadVideo()` RPC handler
  - [ ] Receive chunked file upload
  - [ ] Save to user directory
  - [ ] Return file_id to frontend
- [ ] Update `VideoContext` to track multiple files by file_id
- [ ] Test file upload with large video files (>100MB)
- [x] Add CLI helper (`scripts/upload_video.py`) to perform streaming uploads during manual testing
- [x] Implement `VideoRegistrar` service to register local files + metadata without streaming uploads
- [x] Allow swapping registry persistence backends (JSON today, DB later) via `VideoRegistryStore`
- [x] Add `storage_paths.py` helper to centralize storage root/videos/outputs paths
- [x] Expose `RegisterLocalVideo` gRPC RPC to call the registrar from Tauri
- [ ] **Deliverable:** Frontend can upload videos, backend saves them

#### ✅ **Phase 2: Agent Response Format (1-2 days)**
**Goal:** Agents return structured JSON instead of saving to files

- [ ] Modify `video_to_transcript` (`agents/transcription_agent.py:68-153`)
  - [ ] Add `file_id` parameter
  - [ ] Get file path from `FileStorage.get_file_path(file_id)`
  - [ ] Remove file saving logic (lines 134-140)
  - [ ] Return structured dict:
    ```python
    return {
        "success": True,
        "transcript_text": text,
        "segments": segments,
        "metadata": {"duration": dur, "language": lang}
    }
    ```
- [ ] Modify `detect_objects_in_video` (`agents/vision_agent.py:68-135`)
  - [ ] Add `file_id` parameter
  - [ ] Get file path from `FileStorage.get_file_path(file_id)`
  - [ ] Return structured dict with detections
- [ ] **Deliverable:** Agents return JSON data that frontend can display

#### ✅ **Phase 3: Chat Interface (2-3 days)**
**Goal:** Tauri chat UI can send messages and get responses

- [ ] Implement `SendChatMessage()` RPC handler
  - [ ] Accept message and optional file_id
  - [ ] Pass file_id to orchestrator
  - [ ] Stream responses back to frontend (for progress updates)
  - [ ] Return structured results (transcripts, detections, etc.)
- [ ] Update orchestrator to accept file_id parameter
  - [ ] Modify `process_message(message, file_id)`
  - [ ] Pass file_id to agent tools
- [ ] Add progress streaming support
  - [ ] Yield progress updates during long operations
  - [ ] Frontend can show "Transcribing video..." status
- [ ] **Deliverable:** End-to-end chat flow works: upload → ask question → get answer

---

## Full Implementation Roadmap

### Phase 4: Chat History Persistence (Lower Priority - Week 2)

**Goal:** Save conversations to database, restore on app restart

#### 4.1 Chat History Database
- [ ] Create `services/chat_history.py`
  - [ ] Implement `ChatHistoryStore` class
  - [ ] SQLite database (stored in user app data directory)
  - [ ] Schema:
    ```sql
    CREATE TABLE messages (
      id INTEGER PRIMARY KEY,
      role TEXT,  -- user/assistant
      content TEXT,
      metadata TEXT,  -- JSON with structured results
      timestamp DATETIME
    );
    ```
  - [ ] `save_message(message)` method
  - [ ] `get_history(limit)` method
  - [ ] `clear_history()` method

#### 4.2 gRPC History Endpoint
- [ ] Implement `GetChatHistory()` RPC handler
- [ ] Return conversation history to frontend
- [ ] Frontend restores chat UI on app launch

#### 4.3 Auto-save Chat Messages
- [ ] Save user messages when received
- [ ] Save assistant responses when sent
- [ ] Store structured results (transcripts, detections) in metadata

### Phase 5: Multi-File Management (Lower Priority - Week 2-3)

**Goal:** Track multiple uploaded videos, switch between them

#### 5.1 Video Library
- [ ] Create `services/video_library.py`
  - [ ] Track all uploaded videos with metadata
  - [ ] `list_videos()` → [{file_id, filename, upload_date, ...}]
  - [ ] `get_video_info(file_id)` → metadata
  - [ ] `delete_video(file_id)` → cleanup files
- [ ] Implement `GetUploadedVideos()` RPC
- [ ] Add to protobuf definition

#### 5.2 Update VideoContext
- [ ] Support multiple active videos
- [ ] `set_active_video(file_id)` method
- [ ] Track which video is currently being discussed

### Phase 6: App State Persistence (Lower Priority - Week 3)

**Goal:** Remember app state across restarts (last video, settings, etc.)

- [ ] Create `services/app_state.py`
  - [ ] Save state to JSON file on app close
  - [ ] Restore state on app start
  - [ ] Track: active video, uploaded files, settings
- [ ] Store state in OS-appropriate app data directory:
  - Windows: `C:\Users\{user}\AppData\Local\VideoAnalyzer\`
  - macOS: `~/Library/Application Support/VideoAnalyzer/`
  - Linux: `~/.local/share/VideoAnalyzer/`

### Phase 7: Progress Streaming (Nice-to-Have - Week 3-4)

**Goal:** Real-time progress updates for long operations

- [ ] Add progress callbacks to agent tools
- [ ] Transcription: "Processing chunk 1/10..."
- [ ] Object detection: "Analyzing frame 45/300..."
- [ ] Stream updates via gRPC ChatResponse (PROGRESS type)

### Phase 8: Error Handling & Polish (Week 4)

- [ ] Structured error responses in all RPCs
- [ ] File cleanup for failed uploads
- [ ] Graceful handling of corrupted video files
- [ ] Validate file formats before processing
- [ ] Add logging throughout backend

---

## Key Decision Points

### Q: Where to store uploaded videos?
**Decision:**
- Use OS-appropriate user directories:
  - Windows: `C:\Users\{user}\Documents\VideoAnalyzer\videos\`
  - macOS: `~/Documents/VideoAnalyzer/videos/`
  - Linux: `~/Documents/VideoAnalyzer/videos/`
- Track files with unique file_id (UUID)
- Allow user to delete old videos

### Q: Where to store chat history?
**Decision:**
- SQLite database in app data directory:
  - Windows: `C:\Users\{user}\AppData\Local\VideoAnalyzer\chat.db`
  - macOS: `~/Library/Application Support/VideoAnalyzer/chat.db`
  - Linux: `~/.local/share/VideoAnalyzer/chat.db`
- Single table for messages (simple schema)
- Store structured results as JSON in metadata column

### Q: Do we need session management for desktop app?
**Decision:**
- **NO** - Single user, one app instance
- Global VideoContext singleton is FINE
- Just need to persist state to disk for app restarts
- No need for Redis, session timeouts, concurrent session handling

### Q: Should agents stream progress updates?
**Decision:**
- **YES** - Transcription and object detection are slow (30+ seconds)
- Use gRPC streaming responses
- Stream progress updates: "Transcribing... 45%"
- Frontend shows loading indicator with real-time status

### Q: File upload strategy?
**Decision:**
- Use gRPC streaming for large video files
- Chunk size: 1MB per chunk
- Show upload progress in frontend
- Save to user Documents folder with unique file_id

### Q: How to handle multiple videos?
**Decision:**
- Track all uploaded videos with file_id
- User can switch between videos in chat
- Reference video by file_id: "Analyze the video I just uploaded" → use most recent
- Advanced: "Compare video 1 and video 2" → track multiple active videos

---

## Dependencies to Add

```bash
# gRPC dependencies (REQUIRED for Phase 0)
pip install grpcio grpcio-tools

# Already have these (no new deps needed):
# - opencv-python (for video processing)
# - whisper (for transcription)
# - langchain/langgraph (for orchestration)
```

---

## Summary of File Changes

### Files to Modify (Critical Path)

#### `agents/transcription_agent.py` (Phase 2)
- [ ] Add `file_id` parameter to `video_to_transcript`
- [ ] Replace `video_context.get_current_video_path()` with `FileStorage.get_file_path(file_id)`
- [ ] Remove file saving logic (lines 134-140)
- [ ] Return structured dict instead of string

#### `agents/vision_agent.py` (Phase 2)
- [ ] Add `file_id` parameter to `detect_objects_in_video`
- [ ] Replace `video_context.get_current_video_path()` with `FileStorage.get_file_path(file_id)`
- [ ] Return structured dict with detections
- [ ] Optional: Add progress callback for streaming

#### `orchestrator.py` (Phase 3)
- [ ] Add `file_id` parameter to `process_message()`
- [ ] Pass file_id to agent tools
- [ ] Optional: Return generator for streaming responses

#### `context/video_context.py` (Phase 1)
- [ ] Add `add_video(file_id, file_path)` method
- [ ] Track multiple videos (dict of file_id → path)
- [ ] Keep current singleton pattern (it's fine!)

### Files to Create (Critical Path)

#### Phase 0: gRPC Bootstrap
- [ ] `protos/video_analyzer.proto` (protobuf definition)
- [ ] `grpc_service/__init__.py`
- [ ] `grpc_service/server.py` (minimal server)
- [ ] `grpc_service/video_analyzer_service.py` (service implementation)

#### Phase 1: File Upload
- [ ] `services/__init__.py`
- [ ] `services/file_storage.py` (OS-appropriate file handling)

#### Phase 4: Chat History (Lower Priority)
- [ ] `services/chat_history.py` (SQLite database)

#### Phase 5-6: Additional Features (Lower Priority)
- [ ] `services/video_library.py`
- [ ] `services/app_state.py`

### Configuration Files
- [ ] Update `requirements.txt` with gRPC dependencies
- [ ] Create `README.md` section for running gRPC server
- [ ] Optional: `.env` for gRPC port configuration

---

## Implementation Reference Code

### Minimal Protobuf for MVP (Phase 0)

```protobuf
// protos/video_analyzer.proto
syntax = "proto3";

package video_analyzer;

service VideoAnalyzerService {
  // Phase 1: File upload
  rpc UploadVideo(stream VideoChunk) returns (UploadResponse);

  // Phase 3: Chat interface
  rpc SendChatMessage(ChatRequest) returns (stream ChatResponse);

  // Phase 4: History (lower priority)
  rpc GetChatHistory(HistoryRequest) returns (ChatHistoryResponse);
}

// File upload messages
message VideoChunk {
  bytes data = 1;
  string filename = 2;
  int32 chunk_index = 3;
}

message UploadResponse {
  string file_id = 1;
  bool success = 2;
  string error = 3;
}

// Chat messages
message ChatRequest {
  string message = 1;
  string file_id = 2;  // Optional: which video to analyze
}

message ChatResponse {
  enum ResponseType {
    MESSAGE = 0;
    PROGRESS = 1;
    RESULT = 2;
    ERROR = 3;
  }

  ResponseType type = 1;
  string content = 2;
  string agent_name = 3;
  string result_json = 4;  // Structured data (transcripts, detections)
}

// History messages (Phase 4)
message HistoryRequest {
  int32 limit = 1;
}

message ChatHistoryResponse {
  repeated ChatMessage messages = 1;
}

message ChatMessage {
  string role = 1;
  string content = 2;
  int64 timestamp = 3;
}
```

### FileStorage Implementation (Phase 1)

```python
# services/file_storage.py
import os
import uuid
import platform
from pathlib import Path
from typing import Tuple, Optional

class FileStorage:
    """Manages video file storage in OS-appropriate directories"""

    def __init__(self):
        self.base_dir = self._get_user_video_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}  # file_id -> file_path (in-memory for now)

    def _get_user_video_dir(self) -> Path:
        """Get OS-appropriate directory for storing videos"""
        system = platform.system()

        if system == "Windows":
            base = Path.home() / "Documents" / "VideoAnalyzer" / "videos"
        elif system == "Darwin":  # macOS
            base = Path.home() / "Documents" / "VideoAnalyzer" / "videos"
        else:  # Linux
            base = Path.home() / "Documents" / "VideoAnalyzer" / "videos"

        return base

    def save_uploaded_file(self, file_data: bytes, filename: str) -> Tuple[str, str]:
        """Save uploaded file and return (file_id, file_path)"""
        file_id = uuid.uuid4().hex

        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        file_path = self.base_dir / f"{file_id}_{safe_filename}"

        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_data)

        # Track in memory
        self.files[file_id] = str(file_path)

        return file_id, str(file_path)

    def get_file_path(self, file_id: str) -> str:
        """Get file path by file_id"""
        if file_id not in self.files:
            raise FileNotFoundError(f"File {file_id} not found")
        return self.files[file_id]

    def delete_file(self, file_id: str):
        """Delete a video file"""
        if file_id in self.files:
            file_path = Path(self.files[file_id])
            if file_path.exists():
                file_path.unlink()
            del self.files[file_id]
```

### Minimal gRPC Server (Phase 0)

```python
# grpc_service/server.py
import grpc
from concurrent import futures
import signal
import sys

# Import generated protobuf code
from . import video_analyzer_pb2_grpc
from .video_analyzer_service import VideoAnalyzerServicer

def serve(port: int = 50051):
    """Start gRPC server"""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )

    # Register service
    video_analyzer_pb2_grpc.add_VideoAnalyzerServiceServicer_to_server(
        VideoAnalyzerServicer(), server
    )

    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"✅ gRPC server started on port {port}")
    print(f"   Tauri frontend can connect to: localhost:{port}")

    # Graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down gRPC server...")
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

### gRPC Service Implementation (Phase 1-3)

```python
# grpc_service/video_analyzer_service.py
import grpc
import json
from . import video_analyzer_pb2, video_analyzer_pb2_grpc
from services.file_storage import FileStorage
from context.video_context import get_video_context
from orchestrator import Orchestrator  # Your existing orchestrator

class VideoAnalyzerServicer(video_analyzer_pb2_grpc.VideoAnalyzerServiceServicer):
    """gRPC service for video analysis"""

    def __init__(self):
        self.file_storage = FileStorage()
        self.video_context = get_video_context()
        self.orchestrator = Orchestrator()

    def UploadVideo(self, request_iterator, context):
        """Handle chunked video upload (Phase 1)"""
        try:
            chunks = []
            filename = None

            # Collect all chunks
            for chunk in request_iterator:
                chunks.append(chunk.data)
                if not filename:
                    filename = chunk.filename

            # Save file
            file_data = b''.join(chunks)
            file_id, file_path = self.file_storage.save_uploaded_file(
                file_data, filename
            )

            # Update video context
            self.video_context.set_current_video(file_path)

            print(f"✅ Uploaded video: {filename} → {file_id}")

            return video_analyzer_pb2.UploadResponse(
                file_id=file_id,
                success=True
            )

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return video_analyzer_pb2.UploadResponse(
                success=False,
                error=str(e)
            )

    def SendChatMessage(self, request, context):
        """Handle chat messages with streaming responses (Phase 3)"""
        message = request.message
        file_id = request.file_id or None

        try:
            # If file_id specified, set as current video
            if file_id:
                file_path = self.file_storage.get_file_path(file_id)
                self.video_context.set_current_video(file_path)

            print(f"💬 User: {message}")

            # Process with orchestrator
            # TODO: Update orchestrator to yield streaming responses
            result = self.orchestrator.process_message(message)

            # For now, return single result
            # Later: yield multiple progress updates
            yield video_analyzer_pb2.ChatResponse(
                type=video_analyzer_pb2.ChatResponse.RESULT,
                content=result.get("content", ""),
                agent_name=result.get("agent", ""),
                result_json=json.dumps(result.get("data", {}))
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            yield video_analyzer_pb2.ChatResponse(
                type=video_analyzer_pb2.ChatResponse.ERROR,
                content=f"Error: {str(e)}"
            )

    def GetChatHistory(self, request, context):
        """Retrieve chat history (Phase 4 - lower priority)"""
        # TODO: Implement chat history
        return video_analyzer_pb2.ChatHistoryResponse(messages=[])
```

### Updated Agent Tool Example (Phase 2)

```python
# agents/transcription_agent.py (updated)

from services.file_storage import FileStorage

# Initialize file storage
file_storage = FileStorage()

@tool
def video_to_transcript(file_id: str) -> dict:
    """
    Extract audio from video and transcribe using Whisper.

    Args:
        file_id: ID of the uploaded video file

    Returns:
        Dict with transcript_text, segments, and metadata
    """
    try:
        # Get file path from storage
        video_path = file_storage.get_file_path(file_id)

        if not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"Video file not found: {file_id}"
            }

        # Extract audio (existing logic)
        video = mp.VideoFileClip(video_path)
        audio_path = f"/tmp/{file_id}_audio.wav"
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')

        # Transcribe (existing logic)
        result = whisper_model.transcribe(audio_path)

        # Format segments (existing logic)
        formatted_segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }
            for seg in result["segments"]
        ]

        # CHANGE: Return structured data instead of saving to file
        return {
            "success": True,
            "transcript_text": result["text"],
            "segments": formatted_segments,
            "metadata": {
                "language": result.get("language", "unknown"),
                "duration": video.duration
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        # Cleanup temp audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
```

---

## Testing Checklist

### Phase 0: gRPC Server
- [ ] Install grpcio dependencies
- [ ] Generate protobuf code successfully
- [ ] Server starts without errors
- [ ] Test with grpcurl:
  ```bash
  grpcurl -plaintext localhost:50051 list
  ```

### Phase 1: File Upload
- [ ] Upload small video (10MB)
- [ ] Upload large video (500MB)
- [ ] Verify file saved to correct user directory
- [ ] Check file_id returned correctly
- [ ] Test with corrupted file upload

### Phase 2: Agent Responses
- [ ] Call transcription agent with file_id
- [ ] Verify structured JSON response
- [ ] Call vision agent with file_id
- [ ] Verify detections returned as JSON

### Phase 3: Chat Interface
- [ ] Send chat message without file_id
- [ ] Send chat message with file_id
- [ ] Verify streaming responses work
- [ ] Test error handling (invalid file_id)

### Phase 4: Chat History
- [ ] Save message to database
- [ ] Retrieve history after app restart
- [ ] Clear history works

---

## Next Steps for Frontend Integration

### 🎯 Start Here (Critical Path)

1. **Week 1: Minimal gRPC (Phase 0 + Phase 1)**
   - [ ] Set up protobuf and gRPC server
   - [ ] Implement file upload endpoint
   - [ ] Test with Tauri team
   - **Deliverable:** Tauri can upload a video file

2. **Week 1-2: Agent Integration (Phase 2)**
   - [ ] Update agents to return structured data
   - [ ] Add file_id parameters
   - [ ] Test agents return correct JSON
   - **Deliverable:** Backend can process videos and return results

3. **Week 2: Chat Flow (Phase 3)**
   - [ ] Implement SendChatMessage RPC
   - [ ] Wire up orchestrator
   - [ ] Test end-to-end: upload → chat → result
   - **Deliverable:** Full chat interface works

4. **Week 3-4: Polish (Phase 4-8)**
   - [ ] Add chat history
   - [ ] Add multi-file tracking
   - [ ] Progress streaming
   - [ ] Error handling

---

## Architecture Diagram (Desktop App)

```
┌─────────────────────────────────────┐
│   Tauri Desktop App (Frontend)     │
│   - React UI                        │
│   - File upload component           │
│   - Chat interface                  │
│   - History view                    │
└──────────────┬──────────────────────┘
               │ gRPC
               ▼
┌─────────────────────────────────────┐
│   Python Backend (gRPC Server)      │
│                                     │
│   ┌─────────────────────────────┐  │
│   │  VideoAnalyzerService       │  │
│   │  - UploadVideo()            │  │
│   │  - SendChatMessage()        │  │
│   │  - GetChatHistory()         │  │
│   └─────────────┬───────────────┘  │
│                 │                   │
│   ┌─────────────▼───────────────┐  │
│   │  FileStorage                │  │
│   │  ~/Documents/VideoAnalyzer/ │  │
│   └─────────────────────────────┘  │
│                                     │
│   ┌─────────────────────────────┐  │
│   │  Orchestrator               │  │
│   │  - Intent classification     │  │
│   │  - Agent routing            │  │
│   └─────────────┬───────────────┘  │
│                 │                   │
│   ┌─────────────▼───────────────┐  │
│   │  Agents                     │  │
│   │  - Transcription Agent      │  │
│   │  - Vision Agent             │  │
│   │  - Analysis Agent           │  │
│   └─────────────────────────────┘  │
│                                     │
│   ┌─────────────────────────────┐  │
│   │  ChatHistoryStore (SQLite)  │  │
│   │  ~/.../VideoAnalyzer/chat.db│  │
│   └─────────────────────────────┘  │
└─────────────────────────────────────┘

Storage Locations:
├─ Videos: ~/Documents/VideoAnalyzer/videos/
├─ Database: ~/.../VideoAnalyzer/chat.db
└─ App State: ~/.../VideoAnalyzer/state.json
```

---

**Priority:** HIGH - Frontend integration starts Week 1
**Estimated Timeline:**
- Phase 0-3 (MVP): 2 weeks
- Phase 4-8 (Polish): 2 weeks
**Status:** Ready to implement Phase 0

## Current Focus

**👉 START WITH PHASE 0:** Set up minimal gRPC server so Tauri team can begin integration testing.
