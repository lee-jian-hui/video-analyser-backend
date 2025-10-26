"""
Task Models

Pydantic models for different types of tasks that can be processed
by the multi-agent system. These models provide type safety,
validation, and clear interfaces for task definitions.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal, Union
from pathlib import Path
import os


class BaseTask(BaseModel):
    """Base task model with common fields"""

    description: str = Field(..., description="Description of the task to be performed")
    priority: int = Field(default=1, ge=1, le=5, description="Task priority (1=lowest, 5=highest)")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata for the task")

    class Config:
        extra = "forbid"  # Prevent extra fields


class VideoTask(BaseTask):
    """Task model for video processing operations"""

    file_path: str = Field(..., description="Path to the video file to be processed")
    task_type: Literal["object_detection", "text_extraction", "comprehensive_analysis"] = Field(
        default="object_detection",
        description="Type of video analysis to perform"
    )
    output_format: Literal["summary", "detailed", "json"] = Field(
        default="summary",
        description="Format for the analysis output"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for object detection"
    )
    sample_interval: Optional[int] = Field(
        default=30,
        ge=1,
        description="Interval in seconds for sampling video frames"
    )

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate that the file path exists and is a video file"""
        if not v:
            raise ValueError("File path cannot be empty")

        # Convert to Path object for easier handling
        path = Path(v)

        # Check if it's an absolute path or relative path
        if not path.is_absolute():
            # For relative paths, make them relative to current working directory
            path = Path.cwd() / path

        # Note: We don't check if file exists here as it might be created later
        # or be on a remote system. File existence should be checked at runtime.

        # Check file extension for video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if path.suffix.lower() not in video_extensions:
            raise ValueError(f"File must be a video file. Supported formats: {', '.join(video_extensions)}")

        return str(path)

    def get_task_description(self) -> str:
        """Generate a natural language task description"""
        # Use the description directly instead of predefined mappings
        # This allows for more flexible, natural language task descriptions
        return self.description

    def get_required_tools(self) -> List[str]:
        """Get list of tools required for this task type - deprecated, let LLM decide"""
        # Return empty list to let the LLM dynamically select tools based on task description
        # This removes rigid tool-to-task mappings and allows for more flexibility
        return []


class ImageTask(BaseTask):
    """Task model for image processing operations"""

    file_path: str = Field(..., description="Path to the image file to be processed")
    task_type: Literal["object_detection", "text_extraction", "captioning"] = Field(
        default="object_detection",
        description="Type of image analysis to perform"
    )
    output_format: Literal["summary", "detailed", "json"] = Field(
        default="summary",
        description="Format for the analysis output"
    )

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate that the file path exists and is an image file"""
        if not v:
            raise ValueError("File path cannot be empty")

        path = Path(v)
        if not path.is_absolute():
            path = Path.cwd() / path

        # Check file extension for image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
        if path.suffix.lower() not in image_extensions:
            raise ValueError(f"File must be an image file. Supported formats: {', '.join(image_extensions)}")

        return str(path)


class TextTask(BaseTask):
    """Task model for text processing operations"""

    content: str = Field(..., description="Text content to be processed")
    task_type: Literal["analysis", "summarization", "extraction"] = Field(
        default="analysis",
        description="Type of text processing to perform"
    )
    language: str = Field(default="en", description="Language of the text content")


# Union type for all supported task types
TaskModel = Union[VideoTask, ImageTask, TextTask]


class TaskRequest(BaseModel):
    """Request wrapper for tasks with additional context"""

    task: TaskModel = Field(..., description="The task to be processed")
    execution_mode: Literal["single", "chain", "parallel"] = Field(
        default="chain",
        description="Execution mode for the task"
    )
    agent_preferences: Optional[List[str]] = Field(
        default=None,
        description="Preferred agents for task execution"
    )
    timeout: Optional[int] = Field(
        default=300,
        ge=1,
        description="Timeout in seconds for task execution"
    )

    def get_task_type(self) -> str:
        """Get the task type string"""
        return type(self.task).__name__.lower().replace('task', '')
