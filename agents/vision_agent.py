from typing import Dict, Any, List
from .base_agent import BaseAgent
from graph import MessagesState
from langchain.messages import AIMessage
from langchain.tools import tool
from llm import get_model
from utils.tool_discovery import ToolDiscovery
from configs import Config


# Define tools for video processing
@tool
def detect_objects_in_video(video_path: str, confidence_threshold: float = 0.5, model_size: str = "yolov8n") -> str:
    """Detect objects in video using local YOLO model"""
    try:
        from ultralytics import YOLO
        import cv2

        # Load YOLO model (downloads on first use)
        model = YOLO(f'{model_size}.pt')

        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        all_detections = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on frame
            results = model(frame, conf=confidence_threshold, verbose=False)

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
        return f"Video analysis complete. {len(all_detections)} detections across {frame_count} frames. Detected classes: {list(unique_classes)}"

    except ImportError:
        return "YOLO not installed. Run: pip install ultralytics"
    except Exception as e:
        return f"Error processing video: {str(e)}"

@tool
def extract_text_from_video(video_path: str, sample_interval: int = 30, language: str = "eng") -> str:
    """Extract text from video frames using OCR"""
    try:
        import pytesseract
        from PIL import Image
        import cv2

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_interval)  # Sample every N seconds

        extracted_texts = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames at intervals
            if frame_num % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # Extract text
                text = pytesseract.image_to_string(image, lang=language)
                if text.strip():
                    extracted_texts.append({
                        "timestamp": frame_num / fps,
                        "text": text.strip()
                    })

            frame_num += 1

        cap.release()

        return f"Extracted text from {len(extracted_texts)} frames: {extracted_texts}"

    except ImportError:
        return "Tesseract not installed. Run: pip install pytesseract"
    except Exception as e:
        return f"Error extracting text from video: {str(e)}"



class VisionAgent(BaseAgent):
    """Agent for vision tasks: object recognition, captioning, text/graph extraction"""

    def __init__(self):
        super().__init__(
            name="vision_agent",
            capabilities=["object_recognition", "image_captioning", "text_extraction", "graph_extraction", "ocr"]
        )
        self.model = get_model(Config.GEMINI_API_KEY)
        # Define tools directly for this agent
        self.tools = [detect_objects_in_video, extract_text_from_video]

        # Fallback to discovery if tools list is empty
        if not self.tools:
            self.tools = ToolDiscovery.discover_tools_in_class(self)

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task"""
        task_type = task.get("task_type", "").lower()
        return task_type in ["vision", "image", "ocr", "object_recognition", "captioning"]

    def get_model(self):
        """Get the model instance for this agent"""
        return self.model

    def _process_with_tools(self, state: MessagesState, model_with_tools, tools_by_name, execution_mode: str) -> MessagesState:
        """Process vision-related tasks with tools"""
        from langchain.messages import HumanMessage, ToolMessage
        from utils.logger import get_logger

        logger = get_logger(__name__)
        content = state["messages"][-1].content

        # Get available tool names and descriptions
        available_tools = [tool.name for tool in self.get_tools()]
        tool_descriptions = {tool.name: tool.description for tool in self.get_tools()}

        prompt = f"""You are a vision analysis agent with access to these tools:
{chr(10).join([f"- {name}: {desc}" for name, desc in tool_descriptions.items()])}

Task: {content}

Use the appropriate tools to handle the vision task. Call the relevant tool functions directly.
"""

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
                                error_message = ToolMessage(
                                    content=f"Error executing {tool_name}: {str(tool_error)}",
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
            error_message = AIMessage(content=f"Vision Agent Error: {str(e)}")
            return {
                "messages": state["messages"] + [error_message],
                "llm_calls": state.get("llm_calls", 0)
            }

    def get_tools(self) -> List[Any]:
        """Return actual LangChain tools"""
        return self.tools