"""
gRPC Server for Video Analyzer

Updated to support:
- Streaming file upload
- Chat interface with natural language
- Integration with multi-agent orchestrator
- OS-appropriate file storage
"""

import grpc
from concurrent import futures
from protos import video_analyzer_pb2
from protos import video_analyzer_pb2_grpc
import logging
import json

# Import services
from services.file_storage import FileStorage
from services.chat_history_storage import get_chat_history_storage
from services.chat_history_service import ChatHistoryService
from models.chat_history import ChatHistory
from context.video_context import get_video_context

# Import orchestrator
from orchestrator import MultiStageOrchestrator
from models.task_models import TaskRequest, VideoTask
from services.video_registrar import VideoRegistrar


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoAnalyzerService(video_analyzer_pb2_grpc.VideoAnalyzerServiceServicer):
    """gRPC service for video analysis with multi-agent orchestration"""

    def __init__(self):
        logger.info("Initializing VideoAnalyzerService...")

        # Initialize file storage (OS-appropriate directories)
        self.file_storage = FileStorage()

        # Initialize chat history storage
        self.chat_storage = get_chat_history_storage()
        self.chat_history_service = ChatHistoryService(storage=self.chat_storage)

        # Initialize video context (singleton)
        self.video_context = get_video_context()

        # Initialize multi-agent orchestrator
        self.orchestrator = MultiStageOrchestrator()
        self.video_registrar = VideoRegistrar(file_storage=self.file_storage)

        logger.info("✅ VideoAnalyzerService initialized successfully")
        logger.info(f"   File storage: {self.file_storage.base_dir}")

    def UploadVideo(self, request_iterator, context):
        """
        Handle streaming video upload (Phase 1).

        Receives video file in chunks and saves to OS-appropriate directory.
        Returns file_id for future reference.
        """
        try:
            chunks = []
            filename = None
            chunk_count = 0

            logger.info("📥 Receiving video upload...")

            # Collect all chunks
            for chunk in request_iterator:
                chunks.append(chunk.data)
                chunk_count += 1

                if not filename:
                    filename = chunk.filename
                    logger.info(f"   Filename: {filename}")

            # Combine chunks
            file_data = b''.join(chunks)
            total_size_mb = len(file_data) / (1024 * 1024)

            logger.info(f"   Received {chunk_count} chunks, {total_size_mb:.2f} MB total")

            # Save file using FileStorage
            file_id, file_path = self.file_storage.save_uploaded_file(
                file_data, filename
            )

            # Update video context
            self.video_context.set_current_video(file_path)

            # Save as last video in app state
            self.chat_storage.save_app_state({
                "last_video_id": file_id,
                "last_video_path": file_path,
                "last_video_name": filename
            })

            logger.info(f"✅ Upload successful: {filename} → {file_id}")
            logger.info(f"   Saved to: {file_path}")

            return video_analyzer_pb2.UploadResponse(
                file_id=file_id,
                success=True,
                message=f"Video '{filename}' uploaded successfully ({total_size_mb:.2f} MB)"
            )

        except Exception as e:
            logger.error(f"❌ Upload failed: {e}", exc_info=True)
            return video_analyzer_pb2.UploadResponse(
                file_id="",
                success=False,
                message=f"Upload failed: {str(e)}"
            )

    def RegisterLocalVideo(self, request, context):
        """
        Register a local file selected on the frontend without streaming bytes.
        """
        try:
            metadata = self.video_registrar.register_local_file(
                source_path=request.file_path,
                display_name=request.display_name or None,
                copy_file=not request.reference_only,
            )

            # Save as last video in app state
            self.chat_storage.save_app_state({
                "last_video_id": metadata["file_id"],
                "last_video_path": metadata["stored_path"],
                "last_video_name": metadata["display_name"]
            })

            return video_analyzer_pb2.RegisterVideoResponse(
                file_id=metadata["file_id"],
                stored_path=metadata["stored_path"],
                display_name=metadata["display_name"],
                copied=metadata["copied"],
                size_bytes=int(metadata["size_bytes"]),
                registered_at=float(metadata["registered_at"]),
                message="Video registered successfully",
            )

        except FileNotFoundError as e:
            logger.error(f"❌ RegisterLocalVideo failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return video_analyzer_pb2.RegisterVideoResponse(
                file_id="",
                message=str(e),
            )
        except Exception as e:
            logger.error(f"❌ RegisterLocalVideo error: {e}", exc_info=True)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return video_analyzer_pb2.RegisterVideoResponse(
                file_id="",
                message=str(e),
            )

    def SendChatMessage(self, request, context):
        """
        Handle chat messages with streaming responses.

        Automatically saves messages to chat history.
        Context can be provided by frontend for session resumption.
        """
        message = request.message
        file_id = request.file_id or None
        context_str = request.context or ""  # Optional context from frontend

        logger.info(f"💬 Chat message: '{message}'")

        if file_id:
            logger.info(f"   For video: {file_id}")
        if context_str:
            logger.info(f"   With context: {context_str[:100]}...")

        try:
            # Get file path and video info
            file_path = ""
            filename = "Unknown"
            if file_id:
                file_path = self.file_storage.get_file_path(file_id)
                self.video_context.set_current_video(file_path)
                filename = file_id  # Could be improved to get actual filename
                logger.info(f"   Loaded video: {file_path}")

            # Load or create chat history
            history = self.chat_history_service.load(file_id) if file_id else None
            if not history and file_id:
                history = self.chat_history_service.create_new(
                    video_id=file_id,
                    video_path=file_path,
                    display_name=filename
                )
                logger.info(f"   Created new chat history for: {file_id}")
            else:
                logger.info(f"   Loaded existing history: {history.total_messages} messages")

            # Add user message to history
            if history:
                self.chat_history_service.add_message(history, "user", message)

            # Yield initial progress update
            yield video_analyzer_pb2.ChatResponse(
                type=video_analyzer_pb2.ChatResponse.PROGRESS,
                content="Processing your request...",
                agent_name="orchestrator"
            )

            # Build message with context if provided
            full_message = message
            if context_str:
                full_message = f"[Context from previous conversation: {context_str}]\n\nUser message: {message}"

            # Process with multi-agent orchestrator
            task_request = TaskRequest(
                task=VideoTask(
                    description=full_message,
                    file_path=self.video_context.get_current_video_path() or "",
                    task_type=None
                )
            )

            logger.info("🤖 Processing with multi-agent orchestrator...")
            result = self.orchestrator.process_task(task_request)

            logger.info(f"✅ Processing complete")
            logger.info(f"   Agents used: {result.get('selected_agents', [])}")
            logger.info(f"   LLM calls: {result.get('total_llm_calls', 0)}")

            # Get final result
            final_result = result.get("final_result", "No response generated")

            # Add assistant response to history
            if history:
                self.chat_history_service.add_message(history, "assistant", final_result)
                self.chat_history_service.save(history)
                logger.info(f"   Saved to history: {history.total_messages} total messages")

            # Yield final result
            yield video_analyzer_pb2.ChatResponse(
                type=video_analyzer_pb2.ChatResponse.RESULT,
                content=final_result,
                agent_name=", ".join(result.get("selected_agents", [])),
                result_json=json.dumps({
                    "selected_agents": result.get("selected_agents", []),
                    "execution_plans": result.get("execution_plans", {}),
                    "agent_results": result.get("agent_results", {}),
                    "llm_calls": result.get("total_llm_calls", 0)
                })
            )

        except FileNotFoundError as e:
            logger.error(f"❌ File not found: {e}")
            yield video_analyzer_pb2.ChatResponse(
                type=video_analyzer_pb2.ChatResponse.ERROR,
                content=f"Video file not found. Please upload a video first."
            )
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}", exc_info=True)
            yield video_analyzer_pb2.ChatResponse(
                type=video_analyzer_pb2.ChatResponse.ERROR,
                content=f"Error: {str(e)}"
            )

    def GetLastSession(self, request, context):
        """
        Get information about the last session for resumption prompt.

        Returns session info if available, otherwise has_session=False.
        """
        logger.info("📜 GetLastSession called")

        try:
            # Load app state
            app_state = self.chat_storage.load_app_state()

            if not app_state or "last_video_id" not in app_state:
                logger.info("   No previous session found")
                return video_analyzer_pb2.LastSessionResponse(has_session=False)

            video_id = app_state["last_video_id"]
            video_name = app_state.get("last_video_name", "Unknown")
            video_path = app_state.get("last_video_path", "")

            # Load chat history to get message count
            history = self.chat_history_service.load(video_id)
            message_count = history.total_messages if history else 0
            last_updated = history.updated_at if history else app_state.get("last_updated", "")

            logger.info(f"   Found session: {video_name} ({message_count} messages)")

            return video_analyzer_pb2.LastSessionResponse(
                has_session=True,
                video_id=video_id,
                video_name=video_name,
                video_path=video_path,
                message_count=message_count,
                last_updated=last_updated
            )

        except Exception as e:
            logger.error(f"❌ Error getting last session: {e}", exc_info=True)
            return video_analyzer_pb2.LastSessionResponse(has_session=False)

    def GetChatHistory(self, request, context):
        """
        Retrieve chat history for a specific video.

        Can return just summary or include recent messages.
        """
        video_id = request.video_id
        include_messages = request.include_full_messages

        logger.info(f"📜 GetChatHistory called for video: {video_id} (full={include_messages})")

        try:
            # Load history
            history = self.chat_history_service.load(video_id)

            if not history:
                logger.info(f"   No history found for video: {video_id}")
                return video_analyzer_pb2.GetChatHistoryResponse(
                    video_id=video_id,
                    total_messages=0
                )

            # Build response
            response = video_analyzer_pb2.GetChatHistoryResponse(
                video_id=history.video_id,
                video_name=history.display_name,
                conversation_summary=history.conversation_summary,
                total_messages=history.total_messages,
                created_at=history.created_at,
                updated_at=history.updated_at
            )

            # Include recent messages if requested
            if include_messages:
                for msg in history.recent_messages:
                    response.recent_messages.append(
                        video_analyzer_pb2.ChatMessage(
                            role=msg.role,
                            content=msg.content,
                            timestamp=msg.timestamp
                        )
                    )

            logger.info(f"   Returned history: {history.total_messages} total messages, {len(history.recent_messages)} recent")
            return response

        except Exception as e:
            logger.error(f"❌ Error retrieving chat history: {e}", exc_info=True)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return video_analyzer_pb2.GetChatHistoryResponse(video_id=video_id, total_messages=0)

    def ClearChatHistory(self, request, context):
        """
        Clear chat history for a specific video.
        """
        video_id = request.video_id
        logger.info(f"🗑️  ClearChatHistory called for video: {video_id}")

        try:
            success = self.chat_storage.delete_history(video_id)

            if success:
                logger.info(f"   ✅ Cleared history for: {video_id}")
                return video_analyzer_pb2.ClearHistoryResponse(
                    success=True,
                    message=f"Chat history cleared for video {video_id}"
                )
            else:
                logger.info(f"   ⚠️  No history found for: {video_id}")
                return video_analyzer_pb2.ClearHistoryResponse(
                    success=False,
                    message=f"No chat history found for video {video_id}"
                )

        except Exception as e:
            logger.error(f"❌ Error clearing chat history: {e}", exc_info=True)
            return video_analyzer_pb2.ClearHistoryResponse(
                success=False,
                message=f"Error: {str(e)}"
            )


def serve(port: int = 50051):
    """Start gRPC server"""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            # Support large video files (100MB max)
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )

    # Register service
    video_analyzer_pb2_grpc.add_VideoAnalyzerServiceServicer_to_server(
        VideoAnalyzerService(), server
    )

    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)

    logger.info("=" * 60)
    logger.info("🚀 Video Analyzer gRPC Server")
    logger.info("=" * 60)
    logger.info(f"✅ Server started on {listen_addr}")
    logger.info(f"   Tauri frontend can connect to: localhost:{port}")
    logger.info(f"   Protocol: video_analyzer.VideoAnalyzerService")
    logger.info("=" * 60)
    logger.info("Available RPCs:")
    logger.info("  - UploadVideo (streaming)")
    logger.info("  - SendChatMessage (streaming)")
    logger.info("  - GetChatHistory")
    logger.info("=" * 60)

    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down gRPC server...")
        server.stop(grace=5)
        logger.info("✅ Server stopped")


if __name__ == '__main__':
    serve()
