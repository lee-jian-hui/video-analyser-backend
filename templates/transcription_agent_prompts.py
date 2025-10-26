"""
Transcription Agent Prompt Templates

Centralized prompt management for the transcription agent to maintain consistency
and enable easy modification without touching agent logic.
"""


class TranscriptAgentPrompt:
    """Collection of prompt templates for the transcription agent"""

    SYSTEM_PROMPT = """You are an audio transcription agent specialized in speech-to-text conversion.

Available tools:
{available_tools}

Task: {task_content}

CRITICAL: You are a tool-calling agent. You MUST call the provided tools directly. Never generate code, scripts, or manual instructions. Only use tool calls to complete tasks.
"""

    TOOL_EXECUTION_PROMPT = """You are a transcription agent specialized in converting speech to text from video and audio files.

Available tools (function names are self-descriptive):
{tool_descriptions}

Current task: {task_content}
{file_path_context}

IMPORTANT INSTRUCTIONS:
- You MUST use the available tools by calling them directly - DO NOT generate code or bash commands
- DO NOT write Python scripts or installation commands
- DO NOT provide manual instructions
- ONLY call the provided tools with the correct parameters
- The video is already loaded in context, you don't need to specify the file path
- Choose the most appropriate tool based on the descriptive function name and your task

Call the appropriate tools NOW to complete the task. Do not explain what you will do, just call the tools."""

    ERROR_RESPONSE = """I encountered an error while transcribing: {error_message}

Please check:
- Video file is loaded in context
- Video file contains audio
- Audio is clear enough for transcription
- Required dependencies (Whisper, moviepy) are installed

Would you like to try again?"""

    NO_TOOLS_AVAILABLE = """I don't have access to the required tools for transcription.

Available capabilities: {capabilities}
Required tools: {required_tools}

Please ensure the transcription tools are properly configured."""

    SUCCESS_RESPONSE = """Transcription completed successfully!

{transcript_summary}

The full transcript with timestamps has been generated."""

    @classmethod
    def format_system_prompt(cls, available_tools: str, task_content: str) -> str:
        """Format the system prompt with available tools and task content"""
        return cls.SYSTEM_PROMPT.format(
            available_tools=available_tools,
            task_content=task_content
        )

    @classmethod
    def format_tool_execution_prompt(cls, tool_descriptions: str, task_content: str, file_path_context: str = "") -> str:
        """Format the tool execution prompt with tool descriptions and task content"""
        return cls.TOOL_EXECUTION_PROMPT.format(
            tool_descriptions=tool_descriptions,
            task_content=task_content,
            file_path_context=file_path_context
        )

    @classmethod
    def format_error_response(cls, error_message: str) -> str:
        """Format error response with specific error message"""
        return cls.ERROR_RESPONSE.format(error_message=error_message)

    @classmethod
    def format_no_tools_response(cls, capabilities: list, required_tools: list) -> str:
        """Format response when required tools are not available"""
        return cls.NO_TOOLS_AVAILABLE.format(
            capabilities=", ".join(capabilities),
            required_tools=", ".join(required_tools)
        )

    @classmethod
    def format_success_response(cls, transcript_summary: str) -> str:
        """Format success response with transcript summary"""
        return cls.SUCCESS_RESPONSE.format(transcript_summary=transcript_summary)


class TranscriptAgentExamples:
    """Example prompts and responses for the transcription agent"""

    VIDEO_TRANSCRIPTION_EXAMPLE = {
        "task": "Transcribe the video",
        "expected_tools": ["video_to_transcript"],
        "sample_response": "I'll transcribe the audio from the video using Whisper."
    }

    SUBTITLE_GENERATION_EXAMPLE = {
        "task": "Generate subtitles for the video",
        "expected_tools": ["video_to_transcript"],
        "sample_response": "I'll extract and transcribe the audio to generate subtitles."
    }

    SPEECH_EXTRACTION_EXAMPLE = {
        "task": "What was said in the video?",
        "expected_tools": ["video_to_transcript"],
        "sample_response": "I'll transcribe the spoken content from the video."
    }

    LANGUAGE_SPECIFIC_EXAMPLE = {
        "task": "Transcribe the Spanish audio in this video",
        "expected_tools": ["video_to_transcript"],
        "sample_response": "I'll transcribe the Spanish audio using Whisper's language detection."
    }
