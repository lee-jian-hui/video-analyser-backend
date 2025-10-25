from typing import Dict, Any, List
from .base_agent import BaseAgent
from graph import MessagesState
from langchain.messages import AIMessage
from langchain.tools import tool
from llm import get_model
from utils.tool_discovery import ToolDiscovery
import os


# Define tools first
# @tool
# def analyze_image(image_path: str, analysis_type: str = "general", confidence_threshold: float = 0.5) -> str:
#     """Analyze image for objects, text, and visual elements"""
#     # In production, implement with actual vision APIs or libraries
#     return f"Analysis of {image_path}: {analysis_type} analysis with {confidence_threshold} confidence threshold"

# @tool
# def extract_text_ocr(image_path: str, language: str = "en", output_format: str = "plain") -> str:
#     """Extract text from images using OCR"""
#     # In production, implement with OCR libraries like pytesseract, easyocr, or cloud APIs
#     return f"OCR text extraction from {image_path} in {language}, format: {output_format}"

@tool
def generate_caption(image_path: str, style: str = "detailed", focus_areas: List[str] = None) -> str:
    """Generate descriptive caption for image"""
    # In production, implement with vision models or APIs
    focus = focus_areas or []
    return f"Caption for {image_path} in {style} style, focusing on: {', '.join(focus)}"

@tool
def detect_objects(image_path: str, object_types: List[str] = None, return_coordinates: bool = False) -> str:
    """Detect and classify objects in image"""
    # In production, implement with object detection models
    types = object_types or ["all"]
    coords = "with coordinates" if return_coordinates else "without coordinates"
    return f"Object detection in {image_path} for types: {', '.join(types)} {coords}"


class VisionAgent(BaseAgent):
    """Agent for vision tasks: object recognition, captioning, text/graph extraction"""

    def __init__(self):
        super().__init__(
            name="vision_agent",
            capabilities=["object_recognition", "image_captioning", "text_extraction", "graph_extraction", "ocr"]
        )
        self.model = get_model(os.getenv("GEMINI_API_KEY"))
        # Automatically discover tools decorated with @tool in this module
        self.tools = ToolDiscovery.discover_tools_in_class(self)

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task"""
        task_type = task.get("task_type", "").lower()
        return task_type in ["vision", "image", "ocr", "object_recognition", "captioning"]

    def get_model(self):
        """Get the model instance for this agent"""
        return self.model

    def _process_with_tools(self, state: MessagesState, model_with_tools, tools_by_name) -> MessagesState:
        """Process vision-related tasks with tools"""
        content = state["messages"][-1].content

        prompt = f"""You are a vision analysis agent with access to specialized tools:
        - analyze_image: Analyze images for objects, text, and visual elements
        - extract_text_ocr: Extract text from images using OCR
        - generate_caption: Generate descriptive captions for images
        - detect_objects: Detect and classify objects in images

        Task: {content}

        Use the appropriate tools to handle the vision task. If you need to process an image,
        call the relevant tool functions.
        """

        try:
            response = model_with_tools.invoke([{"role": "user", "content": prompt}])

            new_messages = state["messages"] + [
                AIMessage(content=f"Vision Agent: {response.content}")
            ]

            return {
                "messages": new_messages,
                "llm_calls": state.get("llm_calls", 0) + 1
            }

        except Exception as e:
            error_message = AIMessage(content=f"Vision Agent Error: {str(e)}")
            return {
                "messages": state["messages"] + [error_message],
                "llm_calls": state.get("llm_calls", 0)
            }

    def get_tools(self) -> List[Any]:
        """Return actual LangChain tools"""
        return self.tools