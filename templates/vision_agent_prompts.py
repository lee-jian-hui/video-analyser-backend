"""
Vision Agent Prompt Templates

Centralized prompt management for the vision agent to maintain consistency
and enable easy modification without touching agent logic.
"""


class VisionAgentPrompts:
    """Collection of prompt templates for the vision agent"""

    SYSTEM_PROMPT = """You are a vision analysis agent with access to these tools:
{available_tools}

Task: {task_content}

CRITICAL: You are a tool-calling agent. You MUST call the provided tools directly. Never generate code, scripts, or manual instructions. Only use tool calls to complete tasks.
"""

    TOOL_EXECUTION_PROMPT = """You are a vision analysis agent specialized in image and video processing.

Available tools (function names are self-descriptive):
{tool_descriptions}

Current task: {task_content}
{file_path_context}

IMPORTANT INSTRUCTIONS:
- You MUST use the available tools by calling them directly - DO NOT generate code or bash commands
- DO NOT write Python scripts or installation commands
- DO NOT provide manual instructions
- ONLY call the provided tools with the correct parameters
- Use the provided file path when calling tools that require file input
- Choose the most appropriate tool based on the descriptive function name and your task

Call the appropriate tools NOW to complete the task. Do not explain what you will do, just call the tools."""

    ERROR_RESPONSE = """I encountered an error while processing your vision task: {error_message}

Please check:
- File paths are correct and accessible
- Required dependencies are installed
- Task parameters are valid

Would you like to try again with different parameters?"""

    NO_TOOLS_AVAILABLE = """I don't have access to the required tools for this vision task.

Available capabilities: {capabilities}
Required tools: {required_tools}

Please ensure the necessary tools are properly configured."""

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


class VisionAgentExamples:
    """Example prompts and responses for the vision agent"""

    VIDEO_ANALYSIS_EXAMPLE = {
        "task": "Analyze this video file at /path/to/meeting.mp4 - detect all objects",
        "expected_tools": ["detect_objects_in_video"],
        "sample_response": "I'll analyze the video file to detect objects using the detect_objects_in_video tool."
    }

    IMAGE_ANALYSIS_EXAMPLE = {
        "task": "Extract text from this image file at /path/to/document.png",
        "expected_tools": ["extract_text_from_video"],  # Can be used for images too
        "sample_response": "I'll extract text from the image using OCR capabilities."
    }

    MULTI_STEP_EXAMPLE = {
        "task": "Analyze video for objects and extract any visible text",
        "expected_tools": ["detect_objects_in_video", "extract_text_from_video"],
        "sample_response": "I'll first detect objects in the video, then extract any visible text."
    }