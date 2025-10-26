from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from typing import List, Dict, Any


class OrchestratorPrompts:
    """Centralized prompt templates for orchestrator"""

    # Agent Selection Template
    AGENT_SELECTOR = ChatPromptTemplate.from_template("""
You are an orchestration AI that analyzes user requests and selects appropriate agents.

Available agents:
{available_agents}

Agent capabilities:
{agent_capabilities}

User request: {user_request}

Task: Select which agents are needed for this request. Consider:
- Vision tasks (video/image analysis) → vision_agent
- Audio tasks (transcription, speech) → transcription_agent
- Document creation (PDF, PPT, reports) → generation_agent

Respond with ONLY a JSON array of agent names.
Example: ["vision_agent", "transcription_agent"]
""")

    # Tool Planning Template
    TOOL_PLANNER = ChatPromptTemplate.from_template("""
You are planning tool execution for the {agent_name} agent.

Available tools for {agent_name}:
{tool_names}

Tool descriptions:
{tool_descriptions}

User request: {user_request}
Agent role: {agent_role}

Task: Select tools to execute for this agent in logical order. Consider:
- Dependencies between tools
- Optimal execution sequence
- Task requirements

Respond with ONLY a JSON array of tool names in execution order.
Example: ["detect_objects_in_video", "extract_text_from_video"]
""")

    # Result Aggregation Template
    RESULT_AGGREGATOR = ChatPromptTemplate.from_template("""
You are a result aggregator that combines outputs from multiple AI agents into a comprehensive response.

Original user request: {original_task}

Agent execution results:
{agent_results}

Task: Create a comprehensive summary that:
1. Addresses the user's original request
2. Synthesizes findings from all agents
3. Provides actionable insights
4. Maintains clarity and organization

Format your response as a clear, structured summary.
""")

    # Agent Execution Template (for single/chain mode)
    AGENT_EXECUTION = ChatPromptTemplate.from_template("""
You are the {agent_name} with the following capabilities: {agent_capabilities}

Execution mode: {execution_mode}
{mode_instructions}

Available tools:
{available_tools}

User task: {user_task}

{execution_instructions}
""")

    @staticmethod
    def get_mode_instructions(execution_mode: str) -> str:
        """Get mode-specific instructions"""
        if execution_mode == "single":
            return """
SINGLE MODE: Choose and execute ONLY ONE tool that best addresses the task.
Focus on the most critical aspect of the request.
"""
        else:  # chain mode
            return """
CHAIN MODE: You can execute multiple tools in sequence.
Plan the optimal order considering dependencies and logical flow.
"""

    @staticmethod
    def get_execution_instructions(execution_mode: str) -> str:
        """Get execution-specific instructions"""
        if execution_mode == "single":
            return "Select the single most appropriate tool and execute it."
        else:
            return "Plan and execute a sequence of tools to comprehensively address the task."

    CLARIFICATION_TEMPLATE = """
I can only help with video-analysis workflows handled by the registered desktop agents.

Your latest request was: "{user_request}"

Available agents and focus areas:
{capability_summary}

Please rephrase your question so that it clearly maps to one of these capabilities.
For example:
- "Detect objects in the uploaded video."
- "Generate a transcript for my clip."
- "Summarize what happens in the video I just uploaded."
"""

    @classmethod
    def format_clarification_message(cls, user_request: str, capability_summary: str) -> str:
        summary = capability_summary or "- No agents are currently registered."
        return cls.CLARIFICATION_TEMPLATE.format(
            user_request=user_request,
            capability_summary=summary,
        )


class PromptExamples:
    """Few-shot examples for better LLM performance"""

    AGENT_SELECTION_EXAMPLES = [
        {
            "input": "Analyze this video file and create a summary report",
            "output": '["vision_agent", "generation_agent"]'
        },
        {
            "input": "Transcribe this audio and detect objects in the accompanying video",
            "output": '["transcription_agent", "vision_agent"]'
        },
        {
            "input": "Extract text from video frames",
            "output": '["vision_agent"]'
        }
    ]

    TOOL_PLANNING_EXAMPLES = {
        "vision_agent": [
            {
                "input": "Detect objects and extract text from video",
                "output": '["detect_objects_in_video", "extract_text_from_video"]'
            },
            {
                "input": "Find all people in the video",
                "output": '["detect_objects_in_video"]'
            }
        ]
    }

    @staticmethod
    def create_few_shot_prompt(examples: List[Dict], template: PromptTemplate) -> FewShotPromptTemplate:
        """Create a few-shot prompt template"""
        example_template = PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}"
        )

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="Here are some examples:",
            suffix=template.template,
            input_variables=template.input_variables
        )
