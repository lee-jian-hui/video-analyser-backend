"""
Transcription Agent

Handles speech-to-text transcription tasks using Whisper.
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from llm import get_llm_model
from configs import Config
from langchain.tools import tool

from graph import MessagesState
from templates.transcription_agent_prompts import TranscriptAgentPrompt
from utils.logger import get_logger
from models.agent_capabilities import AgentCapability, CapabilityCategory

import whisper
import moviepy.editor as mp
import tempfile
import os

from context.video_context import get_video_context


# ============================================================================
# AGENT CAPABILITIES DEFINITION
# ============================================================================
TRANSCRIPTION_AGENT_CAPABILITIES = AgentCapability(
    capabilities=[
        "Audio transcription from video",
        "Speech-to-text conversion",
        "Subtitle generation",
        "Spoken word extraction",
        "Video audio analysis",
    ],
    intent_keywords=[
        # Primary keywords
        "transcribe", "transcript", "transcription",
        "speech", "spoken", "audio",
        "subtitle", "subtitles", "captions",
        # Action phrases
        "what said", "what was said", "what they said",
        "convert to text", "extract audio",
        "get transcript", "generate transcript",
        # Related terms
        "voice", "talk", "speaking", "words",
        "dialogue", "conversation",
    ],
    categories=[
        CapabilityCategory.AUDIO,
        CapabilityCategory.TEXT,
    ],
    example_tasks=[
        "Transcribe the video",
        "Generate a transcript for this video",
        "What was said in the video?",
        "Extract all spoken words from the video",
        "Convert the audio to text",
        "Create subtitles for the video",
    ],
    routing_priority=8,  # High priority for clear audio/transcription requests
)


logger = get_logger(__name__)


@tool
def video_to_transcript(video_path: str = None) -> str:
    """Extract audio from video and transcribe to text using Whisper.

    Uses the current video from VideoContext.
    """
    language: str = "en"
    try:
        # Prefer explicit path if provided (supports subprocess execution)
        if not video_path:
            video_context = get_video_context()
            video_path = video_context.get_current_video_path()

        if not video_path:
            return "Error: No video loaded in context. Please load a video first."

        # Check if video file exists
        if not os.path.exists(video_path):
            return f"Error: Video file not found at {video_path}"

        # Load Whisper model using model manager
        from ai_model_manager import get_model_manager
        model_manager = get_model_manager()
        model = model_manager.get_whisper_model("base")

        # Extract audio from video
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            try:
                # Load video and extract audio
                video = mp.VideoFileClip(video_path)
                audio = video.audio

                # Write audio to temporary file
                audio.write_audiofile(temp_audio.name, verbose=False, logger=None)

                # Close video to free resources
                video.close()
                audio.close()

                # Transcribe audio
                result = model.transcribe(temp_audio.name, language=language)

                # Extract text and timestamps
                transcript_text = result["text"]

                # Format with timestamps if segments are available
                if "segments" in result:
                    formatted_transcript = []
                    for segment in result["segments"]:
                        start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
                        end_time = f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}"
                        formatted_transcript.append(f"[{start_time}-{end_time}] {segment['text'].strip()}")

                    # Full transcript for file
                    full_transcript = f"Transcription complete. Full text: {transcript_text}\n\nTimestamped transcript:\n" + "\n".join(formatted_transcript)

                    # Summary for LLM (first 5 and last 5 segments)
                    total_segments = len(formatted_transcript)
                    if total_segments > 10:
                        preview_segments = formatted_transcript[:5] + [f"\n... ({total_segments - 10} more segments) ...\n"] + formatted_transcript[-5:]
                    else:
                        preview_segments = formatted_transcript

                    llm_output = f"Transcription complete. Total segments: {total_segments}\n\nPreview:\n" + "\n".join(preview_segments)
                else:
                    full_transcript = f"Transcription complete: {transcript_text}"
                    llm_output = full_transcript

                # Save FULL transcript to file
                transcript_filename = os.path.splitext(os.path.basename(video_path))[0] + "_transcript.txt"
                transcript_path = os.path.join(os.path.dirname(video_path), transcript_filename)

                try:
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(full_transcript)
                    return f"{llm_output}\n\n✅ Full transcript saved to: {transcript_path}"
                except Exception as save_error:
                    return f"{llm_output}\n\n⚠️ Warning: Could not save transcript file: {str(save_error)}"

            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio.name):
                    os.unlink(temp_audio.name)

    except ImportError as e:

        return f"{str(e)}"
    except Exception as e:
        return f"Error during transcription: {str(e)}"


class TranscriptionAgent(BaseAgent):
    """Agent for speech-to-text transcription tasks"""

    def __init__(self):
        # Use capabilities from the module-level definition
        super().__init__(
            name="transcription_agent",
            capabilities=TRANSCRIPTION_AGENT_CAPABILITIES.capabilities
        )
        self.capability_definition = TRANSCRIPTION_AGENT_CAPABILITIES
        self.model = get_llm_model()

        # Define tools directly for this agent
        self.tools = [video_to_transcript]

        # Fallback to discovery if tools list is empty
        if not self.tools:
            from utils.tool_discovery import ToolDiscovery
            self.tools = ToolDiscovery.discover_tools_in_class(self)

        # Register capabilities with the registry
        from models.agent_capabilities import AgentCapabilityRegistry
        AgentCapabilityRegistry.register(self.name, self.capability_definition)

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task (legacy support)"""
        # Legacy: Check old-style task_type
        task_type = task.get("task_type", "").lower()
        if task_type in ["transcription", "speech_to_text", "audio"]:
            return True

        # New: Check description-based intent matching
        description = task.get("description", "")
        if description:
            return self.capability_definition.matches_description(description)

        return False

    def get_model(self):
        """Get the model instance for this agent"""
        return self.model

    def _process_task_request(self, state: MessagesState, model_with_tools, tools_by_name, task_request, execution_mode: str, planned_tools: Optional[List[str]] = None) -> MessagesState:
        """Process transcription tasks using TaskRequest model"""

        # Extract task information
        task = task_request.task
        task_description = task.get_task_description()

        # Get available tool descriptions
        tool_descriptions = {tool.name: tool.description for tool in self.get_tools()}

        # Format tool descriptions for the prompt
        formatted_tools = "\n".join([f"- {name}: {desc}" for name, desc in tool_descriptions.items()])

        # Add file path context
        file_path_context = f"File to process: {task.file_path}" if hasattr(task, 'file_path') else ""

        # Use template prompt
        prompt = TranscriptAgentPrompt.format_tool_execution_prompt(
            tool_descriptions=formatted_tools,
            task_content=task_description,
            file_path_context=file_path_context
        )

        try:
            # Ensure proper message role alternation for ChatHuggingFace
            # Start fresh with just the task prompt as a user message
            current_messages = [HumanMessage(content=prompt)]
            llm_calls = 0
            max_iterations = 3  # Transcription is usually simpler

            for iteration in range(max_iterations):
                logger.debug(f"Transcription agent iteration {iteration + 1}")

                # Invoke the model
                response = model_with_tools.invoke(current_messages)
                current_messages.append(response)
                llm_calls += 1

                logger.debug(f"Model response: {response.content}")
                logger.debug(f"Tool calls: {getattr(response, 'tool_calls', [])}")

                # Check if the model made tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Process each tool call
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        # # Use the file path from the task if video_path is not provided
                        # if tool_name == "video_to_transcript":
                        #     if "video_path" not in tool_args and hasattr(task, 'file_path'):
                        #         tool_args["video_path"] = task.file_path

                        logger.debug(f"Calling tool: {tool_name} with args: {tool_args}")

                        if tool_name in tools_by_name:
                            try:
                                # Execute the tool
                                tool_result = tools_by_name[tool_name].invoke(tool_args)
                                logger.debug(f"Tool result: {tool_result}")

                                # Add tool result to messages
                                tool_message = ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call["id"]
                                )
                                current_messages.append(tool_message)

                            except Exception as tool_error:
                                logger.error(f"Tool execution error: {tool_error}")
                                error_content = TranscriptAgentPrompt.format_error_response(
                                    error_message=f"Error executing {tool_name}: {str(tool_error)}"
                                )
                                error_message = ToolMessage(
                                    content=error_content,
                                    tool_call_id=tool_call["id"]
                                )
                                current_messages.append(error_message)
                        else:
                            logger.warning(f"Unknown tool: {tool_name}")
                            error_message = ToolMessage(
                                content=f"Unknown tool: {tool_name}",
                                tool_call_id=tool_call["id"]
                            )
                            current_messages.append(error_message)

                    # Continue the conversation if in chain mode
                    if execution_mode == "chain":
                        continue
                    else:
                        break
                else:
                    # No more tool calls, we're done
                    break

            return {
                "messages": current_messages,
                "llm_calls": state.get("llm_calls", 0) + llm_calls
            }

        except Exception as e:
            logger.error(f"Transcription agent processing error: {e}")
            error_content = TranscriptAgentPrompt.format_error_response(
                error_message=f"Transcription agent processing error: {str(e)}"
            )
            error_message = AIMessage(content=error_content)
            return {
                "messages": state["messages"] + [error_message],
                "llm_calls": state.get("llm_calls", 0)
            }

    def get_tools(self) -> List[Any]:
        """Return actual LangChain tools"""
        return self.tools
