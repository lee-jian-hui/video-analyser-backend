from typing import Dict, Any, List
from .base_agent import BaseAgent
from graph import MessagesState
from langchain.messages import AIMessage
from llm import get_model
import os


class TranscriptionAgent(BaseAgent):
    """Agent for speech-to-text transcription tasks"""

    def __init__(self):
        super().__init__(
            name="transcription_agent",
            capabilities=["speech_to_text", "audio_processing", "transcription"]
        )
        self.model = get_model(os.getenv("GEMINI_API_KEY"))

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task"""
        task_type = task.get("task_type", "").lower()
        return task_type in ["transcription", "speech_to_text", "audio"]

    def process(self, state: MessagesState) -> MessagesState:
        """Process audio transcription task"""
        # Extract task content
        content = state["messages"][-1].content

        # For now, simulate transcription with LLM
        # In production, you'd use actual speech-to-text services like:
        # - Google Speech-to-Text
        # - OpenAI Whisper
        # - Azure Speech Services

        prompt = f"""You are a transcription agent.
        Task: {content}

        If this involves audio transcription, provide instructions on how to process the audio file.
        If audio data is provided, simulate a transcription result.
        """

        try:
            response = self.model.invoke([{"role": "user", "content": prompt}])

            # Add the response to messages
            new_messages = state["messages"] + [
                AIMessage(content=f"Transcription Agent: {response.content}")
            ]

            return {
                "messages": new_messages,
                "llm_calls": state.get("llm_calls", 0) + 1
            }

        except Exception as e:
            error_message = AIMessage(content=f"Transcription Agent Error: {str(e)}")
            return {
                "messages": state["messages"] + [error_message],
                "llm_calls": state.get("llm_calls", 0)
            }

    def get_tools(self) -> List[Any]:
        """Return transcription-related tools"""
        return [
            {
                "name": "transcribe_audio",
                "description": "Transcribe audio file to text",
                "parameters": {
                    "audio_file_path": "string",
                    "language": "string (optional)",
                    "format": "string (optional)"
                }
            },
            {
                "name": "process_audio_stream",
                "description": "Process real-time audio stream",
                "parameters": {
                    "stream_url": "string",
                    "chunk_size": "integer (optional)"
                }
            }
        ]