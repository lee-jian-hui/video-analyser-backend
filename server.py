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
        Handle chat messages with streaming responses (Phase 3).

        Processes natural language queries using multi-agent orchestrator.
        Streams progress updates and final results.
        """
        message = request.message
        file_id = request.file_id or None

        logger.info(f"💬 Chat message: '{message}'")

        if file_id:
            logger.info(f"   For video: {file_id}")

        try:
            # If file_id specified, set as current video
            if file_id:
                file_path = self.file_storage.get_file_path(file_id)
                self.video_context.set_current_video(file_path)
                logger.info(f"   Loaded video: {file_path}")

            # Yield initial progress update
            yield video_analyzer_pb2.ChatResponse(
                type=video_analyzer_pb2.ChatResponse.PROGRESS,
                content="Processing your request...",
                agent_name="orchestrator"
            )

            # Process with multi-agent orchestrator
            # Create TaskRequest for the orchestrator
            task_request = TaskRequest(
                task=VideoTask(
                    description=message,
                    file_path=self.video_context.get_current_video_path() or "",
                    task_type=None
                )
            )

            logger.info("🤖 Processing with multi-agent orchestrator...")
            result = self.orchestrator.process_task(task_request)

            logger.info(f"✅ Processing complete")
            logger.info(f"   Agents used: {result.get('selected_agents', [])}")
            logger.info(f"   LLM calls: {result.get('total_llm_calls', 0)}")

            # Yield final result
            yield video_analyzer_pb2.ChatResponse(
                type=video_analyzer_pb2.ChatResponse.RESULT,
                content=result.get("final_result", "No response generated"),
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

    def GetChatHistory(self, request, context):
        """
        Retrieve chat history (Phase 4 - TODO).

        Currently returns empty history. Will be implemented
        with ChatHistoryStore in Phase 4.
        """
        logger.info("📜 GetChatHistory called (not yet implemented)")

        # TODO: Implement ChatHistoryStore in Phase 4
        return video_analyzer_pb2.ChatHistoryResponse(messages=[])


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
