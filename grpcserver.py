# This file contains the gRPC service handlers for the multi-agent system
# Add these methods to your existing gRPC server implementation

from multi_agent_coordinator import MultiAgentCoordinator
from agents.transcription_agent import TranscriptionAgent
from agents.vision_agent import VisionAgent
from agents.generation_agent import GenerationAgent


class MultiAgentService:
    """gRPC service handlers for multi-agent system"""

    def __init__(self):
        # Initialize coordinator and register agents
        self.coordinator = MultiAgentCoordinator()
        self._register_agents()

    def _register_agents(self):
        """Register all available agents"""
        # Register transcription agent
        transcription_agent = TranscriptionAgent()
        self.coordinator.register_agent(transcription_agent)

        # Register vision agent
        vision_agent = VisionAgent()
        self.coordinator.register_agent(vision_agent)

        # Register generation agent
        generation_agent = GenerationAgent()
        self.coordinator.register_agent(generation_agent)

    # Add these methods to your existing gRPC service class:

    def ProcessTask(self, request, context):
        """
        Handle task processing requests from frontend
        Expected request format: {
            "task_type": "transcription|vision|generation",
            "content": "task content or file path",
            "options": {"key": "value"}  # optional parameters
        }
        """
        try:
            # Convert gRPC request to dict
            task_data = {
                "task_type": request.task_type,
                "content": request.content,
                "options": dict(request.options) if hasattr(request, 'options') else {}
            }

            # Process through coordinator
            result = self.coordinator.process_request(task_data)

            # Convert result back to gRPC response
            return YourResponseMessage(
                success=result["success"],
                messages=result.get("messages", []),
                agent_used=result.get("agent_used", ""),
                error=result.get("error", ""),
                llm_calls=result.get("llm_calls", 0)
            )

        except Exception as e:
            return YourResponseMessage(
                success=False,
                error=f"Server error: {str(e)}",
                agent_used="",
                messages=[],
                llm_calls=0
            )

    def GetAvailableAgents(self, request, context):
        """Get list of available agents and their capabilities"""
        try:
            agents = self.coordinator.get_available_agents()

            # Convert to gRPC response format
            agent_list = []
            for name, capabilities in agents.items():
                agent_list.append(YourAgentInfo(
                    name=name,
                    capabilities=capabilities
                ))

            return YourAgentListResponse(agents=agent_list)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting agents: {str(e)}")
            return YourAgentListResponse(agents=[])

    def HealthCheck(self, request, context):
        """Health check for all agents"""
        try:
            status = self.coordinator.health_check()

            return YourHealthResponse(
                overall_status="healthy" if all(
                    agent["status"] == "healthy" for agent in status.values()
                ) else "degraded",
                agent_statuses=status
            )

        except Exception as e:
            return YourHealthResponse(
                overall_status="unhealthy",
                agent_statuses={"error": str(e)}
            )

    def StreamTaskProgress(self, request, context):
        """Stream task progress for long-running operations"""
        try:
            # For streaming responses, you'll need to yield progress updates
            task_data = {
                "task_type": request.task_type,
                "content": request.content,
                "options": dict(request.options) if hasattr(request, 'options') else {}
            }

            # Yield initial progress
            yield YourProgressResponse(
                status="started",
                progress=0.0,
                message="Task initiated"
            )

            # Process task and yield updates
            # This would need to be implemented in each agent to support streaming
            result = self.coordinator.process_request(task_data)

            # Yield final result
            yield YourProgressResponse(
                status="completed" if result["success"] else "failed",
                progress=1.0,
                message="Task completed",
                final_result=result
            )

        except Exception as e:
            yield YourProgressResponse(
                status="error",
                progress=0.0,
                message=f"Error: {str(e)}"
            )


# Notes for integration:
# 1. Replace "YourResponseMessage", "YourAgentInfo", etc. with your actual protobuf message types
# 2. Add proper imports for your protobuf generated classes
# 3. Initialize this service in your existing gRPC server setup
# 4. Make sure to handle authentication/authorization as needed
# 5. Add proper logging and monitoring