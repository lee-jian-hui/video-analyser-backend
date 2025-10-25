from typing import Dict, Any, List
from .base_agent import BaseAgent
from graph import MessagesState
from langchain.messages import AIMessage
from llm import get_model
import os


class GenerationAgent(BaseAgent):
    """Agent for generating PDF and PowerPoint summaries and documents"""

    def __init__(self):
        super().__init__(
            name="generation_agent",
            capabilities=["pdf_generation", "powerpoint_generation", "document_creation", "summary_generation"]
        )
        self.model = get_model(os.getenv("GEMINI_API_KEY"))

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle the task"""
        task_type = task.get("task_type", "").lower()
        return task_type in ["generation", "pdf", "powerpoint", "ppt", "document", "summary"]

    def process(self, state: MessagesState) -> MessagesState:
        """Process document generation tasks"""
        content = state["messages"][-1].content

        # In production, you'd integrate with:
        # - ReportLab for PDF generation
        # - python-pptx for PowerPoint creation
        # - WeasyPrint for HTML to PDF
        # - Jinja2 for templating

        prompt = f"""You are a document generation agent specialized in creating:
        - PDF summaries and reports
        - PowerPoint presentations
        - Structured documents
        - Executive summaries

        Task: {content}

        Generate a structured outline or content for the requested document type.
        Include formatting suggestions, section headers, and content organization.
        """

        try:
            response = self.model.invoke([{"role": "user", "content": prompt}])

            new_messages = state["messages"] + [
                AIMessage(content=f"Generation Agent: {response.content}")
            ]

            return {
                "messages": new_messages,
                "llm_calls": state.get("llm_calls", 0) + 1
            }

        except Exception as e:
            error_message = AIMessage(content=f"Generation Agent Error: {str(e)}")
            return {
                "messages": state["messages"] + [error_message],
                "llm_calls": state.get("llm_calls", 0)
            }

    def get_tools(self) -> List[Any]:
        """Return document generation tools"""
        return [
            {
                "name": "generate_pdf",
                "description": "Generate PDF document from content",
                "parameters": {
                    "content": "string",
                    "template": "string (optional)",
                    "output_path": "string",
                    "format_options": "object (optional)"
                }
            },
            {
                "name": "create_powerpoint",
                "description": "Create PowerPoint presentation",
                "parameters": {
                    "slides_content": "array",
                    "template": "string (optional)",
                    "output_path": "string",
                    "theme": "string (optional)"
                }
            },
            {
                "name": "generate_summary",
                "description": "Generate executive summary from content",
                "parameters": {
                    "source_content": "string",
                    "summary_type": "string (executive|technical|brief)",
                    "max_length": "integer (optional)",
                    "key_points": "array (optional)"
                }
            },
            {
                "name": "format_document",
                "description": "Apply formatting and styling to document",
                "parameters": {
                    "content": "string",
                    "format_type": "string (academic|business|technical)",
                    "include_charts": "boolean (optional)",
                    "output_format": "string (pdf|docx|html)"
                }
            }
        ]