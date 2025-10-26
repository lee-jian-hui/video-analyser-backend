"""
Vision Agent

Handles computer vision tasks including object detection, tracking, and visual analysis.
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from graph import MessagesState
from langchain.messages import AIMessage
from langchain.tools import tool
from llm import get_llm_model
from utils.tool_discovery import ToolDiscovery
from configs import Config
from templates.vision_agent_prompts import VisionAgentPrompts
from langchain.messages import HumanMessage, ToolMessage
from utils.logger import get_logger
from models.agent_capabilities import AgentCapability, CapabilityCategory
from ultralytics import YOLO
import cv2
from context import get_video_context
from storage_paths import get_outputs_dir


# ============================================================================
# AGENT CAPABILITIES DEFINITION
# ============================================================================
VISION_AGENT_CAPABILITIES = AgentCapability(
    capabilities=[
        "Object detection in videos",
        "Visual content analysis",
        "People and animal detection",
        "Vehicle and object tracking",
        "Scene understanding",
    ],
    intent_keywords=[
        # Primary keywords
        "detect", "detection", "identify", "find",
        "locate", "search", "spot",
        # Objects
        "object", "objects", "person", "people",
        "car", "vehicle", "animal", "thing",
        # Actions
        "what see", "what's in", "show me",
        "track", "follow", "movement",
        "analyze video", "video analysis",
        # Visual terms
        "visual", "vision", "image", "frame",
        "appear", "visible", "scene",
    ],
    categories=[
        CapabilityCategory.VISION,
        CapabilityCategory.ANALYSIS,
    ],
    example_tasks=[
        "Detect objects in the video",
        "Find all people in the video",
        "What cars appear in the video?",
        "Identify all animals in the video",
        "Analyze what's happening in the video",
        "Track movement of objects",
    ],
    routing_priority=9,  # High priority for visual/object detection requests
)


# Define tools for video processing
@tool
def detect_objects_in_video() -> str:
    """Detect objects in the current video using local YOLO model.
            
    Returns:
        Detailed summary of detected objects with counts and confidence scores
    """
    confidence_threshold: float = 0.5
    model_size: str = "yolov8n"
    logger = get_logger(__name__)
    try:

        # Load YOLO model using model manager
        from ai_model_manager import get_model_manager
        model_manager = get_model_manager()
        model = model_manager.get_yolo_model(f'{model_size}.pt')

        video_context = get_video_context()
        video_path = video_context.get_current_video_path()
        
        if not video_path:
            return "No video file is currently loaded. Please load a video first."
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        all_detections = []
        frame_num = 0

        project_dir = get_outputs_dir()
        project_dir.mkdir(parents=True, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on frame
            results = model(
                frame,
                conf=confidence_threshold,
                verbose=False,
                save=False,
                project=str(project_dir / "yolo_runs"),
                name="vision_agent",
                exist_ok=True,
            )

            frame_detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        class_name = model.names[class_id]

                        frame_detections.append({
                            "frame": frame_num,
                            "timestamp": frame_num / fps,
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": bbox
                        })

            all_detections.extend(frame_detections)
            frame_num += 1

        cap.release()

        # Summarize detections
        unique_classes = set(det["class"] for det in all_detections)
        logger.info(
            "Vision agent detection complete: %d detections, classes=%s",
            len(all_detections),
            list(unique_classes),
        )
        return f"Video analysis complete. {len(all_detections)} detections across {frame_count} frames. Detected classes: {list(unique_classes)}"

    except ImportError:
        logger.exception("YOLO not installed")
        return "YOLO not installed. Run: pip install ultralytics"
    except Exception as e:
        logger.exception("Error during detect_objects_in_video")
        return f"Error processing video: {str(e)}"


@tool
def dummy():
    pass

# @tool
# def extract_text_from_video(sample_interval: int = 30, language: str = "eng") -> str:
#     """Extract text from the current video frames using OCR"""
#     try:
#         import pytesseract
#         from PIL import Image
#         import cv2

#         # Get video path from context
#         from context import get_video_context
#         video_context = get_video_context()
#         video_path = video_context.get_current_video_path()
        
#         if not video_path:
#             return "No video file is currently loaded. Please load a video first."
        
#         # Open video
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_interval = int(fps * sample_interval)  # Sample every N seconds

#         extracted_texts = []
#         frame_num = 0

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Sample frames at intervals
#             if frame_num % frame_interval == 0:
#                 # Convert BGR to RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image = Image.fromarray(frame_rgb)

#                 # Extract text
#                 text = pytesseract.image_to_string(image, lang=language)
#                 if text.strip():
#                     extracted_texts.append({
#                         "timestamp": frame_num / fps,
#                         "text": text.strip()
#                     })

#             frame_num += 1

#         cap.release()

#         return f"Extracted text from {len(extracted_texts)} frames: {extracted_texts}"

#     except ImportError:
#         return "Tesseract not installed. Run: pip install pytesseract"
#     except Exception as e:
#         return f"Error extracting text from video: {str(e)}"



class VisionAgent(BaseAgent):
    """Agent for vision tasks: object recognition, captioning, text/graph extraction"""

    def __init__(self):
        # Use capabilities from the module-level definition
        super().__init__(
            name="vision_agent",
            capabilities=VISION_AGENT_CAPABILITIES.capabilities
        )
        self.capability_definition = VISION_AGENT_CAPABILITIES
        self.model = get_llm_model()

        # Define tools directly for this agent
        self.tools = [detect_objects_in_video]

        # Fallback to discovery if tools list is empty
        if not self.tools:
            self.tools = ToolDiscovery.discover_tools_in_class(self)

        # Register capabilities with the registry
        from models.agent_capabilities import AgentCapabilityRegistry
        AgentCapabilityRegistry.register(self.name, self.capability_definition)

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task (legacy support)"""
        # Legacy: Check old-style task_type
        task_type = task.get("task_type", "").lower()
        if task_type in ["vision", "image", "ocr", "object_recognition", "captioning", "object_detection"]:
            return True

        # New: Check description-based intent matching
        description = task.get("description", "")
        if description:
            return self.capability_definition.matches_description(description)

        return False

    def get_model(self):
        """Get the model instance for this agent"""
        return self.model

    def _process_with_tools(self, state: MessagesState, model_with_tools, tools_by_name, execution_mode: str, file_path: str = "") -> MessagesState:
        """Process vision-related tasks with tools"""

        logger = get_logger(__name__)
        content = state["messages"][-1].content

        # Get available tool names (function names are descriptive)
        tool_names = [tool.name for tool in self.get_tools()]

        # Format tool names for the prompt - rely on descriptive function names
        formatted_tools = "\n".join([f"- {name}" for name in tool_names])

        # Use template prompt
        prompt = VisionAgentPrompts.format_tool_execution_prompt(
            tool_descriptions=formatted_tools,
            task_content=content
        )

        try:
            # Start with the prompt
            current_messages = state["messages"] + [HumanMessage(content=prompt)]
            llm_calls = 0
            max_iterations = 5  # Prevent infinite loops

            for iteration in range(max_iterations):
                logger.debug(f"Vision agent iteration {iteration + 1}")

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
                                error_content = VisionAgentPrompts.format_error_response(
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
            logger.error(f"Vision agent processing error: {e}")
            error_content = VisionAgentPrompts.format_error_response(
                error_message=f"Vision agent processing error: {str(e)}"
            )
            error_message = AIMessage(content=error_content)
            return {
                "messages": state["messages"] + [error_message],
                "llm_calls": state.get("llm_calls", 0)
            }

    def _process_task_request(self, state: MessagesState, model_with_tools, tools_by_name, task_request, execution_mode: str, planned_tools: Optional[List[str]] = None) -> MessagesState:
        """Process vision tasks using TaskRequest model"""
        from langchain.messages import HumanMessage, ToolMessage
        from utils.logger import get_logger

        logger = get_logger(__name__)

        # Extract task information
        task = task_request.task
        task_description = task.get_task_description()

        # Get available tool names (function names are descriptive)
        tool_names = [tool.name for tool in self.get_tools()]

        # Format tool names for the prompt - rely on descriptive function names
        formatted_tools = "\n".join([f"- {name}" for name in tool_names])

        # Video context is already set by orchestrator, just note it in the prompt
        if hasattr(task, 'file_path'):
            file_path_context = f"Working with video: {task.file_path}"
        else:
            file_path_context = ""

        # Use template prompt
        prompt = VisionAgentPrompts.format_tool_execution_prompt(
            tool_descriptions=formatted_tools,
            task_content=task_description,
            file_path_context=file_path_context
        )

        try:
            # Ensure proper message role alternation for ChatHuggingFace
            # Start fresh with just the task prompt as a user message
            current_messages = [HumanMessage(content=prompt)]
            llm_calls = 0
            max_iterations = 5  # Prevent infinite loops

            for iteration in range(max_iterations):
                logger.debug(f"Vision agent iteration {iteration + 1}")

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

                        # Video path is handled by context now, no need to inject it

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
                                error_content = VisionAgentPrompts.format_error_response(
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
            logger.error(f"Vision agent processing error: {e}")
            error_content = VisionAgentPrompts.format_error_response(
                error_message=f"Vision agent processing error: {str(e)}"
            )
            error_message = AIMessage(content=error_content)
            return {
                "messages": state["messages"] + [error_message],
                "llm_calls": state.get("llm_calls", 0)
            }

    def get_tools(self) -> List[Any]:
        """Return actual LangChain tools"""
        return self.tools
