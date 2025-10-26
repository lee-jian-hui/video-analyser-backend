"""
Refactored Multi-Stage Orchestrator using:
- Pydantic BaseModel for state
- Conditional edges for routing
- Status-based flow control
- Built-in retry logic
"""
import logging
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from llm import get_function_calling_llm, get_chat_llm
from multi_agent_coordinator import MultiAgentCoordinator
from models.task_models import TaskRequest
from models.agent_capabilities import AgentCapabilityRegistry
from templates.orchestrator_prompts import OrchestratorPrompts
from utils.logger import get_logger
from configs import Config
import json
import re


class OrchestratorState(BaseModel):
    """Pydantic state for orchestrator workflow with status-based routing"""

    # ===== FLOW CONTROL =====
    status: Literal[
        "INIT",                 # Starting state
        "SELECT_AGENTS",        # Selecting which agents to use
        "PLAN_TOOLS",          # Planning tool execution for agents
        "EXECUTE_AGENT",       # Executing current agent
        "GENERATE_RESPONSE",   # All agents done, generating response
        "NEED_CLARIFICATION",  # Can't proceed, need user input
        "COMPLETE",            # Success
        "ERROR"                # Failed after retries
    ] = "INIT"

    # ===== RETRY MECHANISM =====
    retry_count: int = 0
    max_retries: int = Field(default=2)
    retry_stage: str = ""  # Which stage needs retry: "selection", "planning", "execution"
    last_error: str = ""

    # ===== TASK DATA =====
    task_request: Any = None  # TaskRequest object
    messages: List[Any] = Field(default_factory=list)

    # ===== AGENT SELECTION =====
    selected_agents: List[str] = Field(default_factory=list)

    # ===== TOOL PLANNING =====
    execution_plans: Dict[str, List[str]] = Field(default_factory=dict)

    # ===== EXECUTION =====
    current_agent_index: int = 0
    agent_results: Dict[str, Any] = Field(default_factory=dict)

    # ===== RESPONSE =====
    chat_response: str = ""
    final_result: str = ""
    clarification_message: str = ""

    # ===== METRICS =====
    planner_llm_calls: int = 0
    agent_llm_calls: int = 0
    chat_llm_calls: int = 0

    class Config:
        arbitrary_types_allowed = True  # For TaskRequest and Message objects


class MultiStageOrchestrator:
    """
    Refactored orchestrator with:
    - Pydantic state model
    - Conditional edge routing
    - Status-based flow control
    - Retry logic
    """

    def __init__(self, agents=None):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing Refactored MultiStageOrchestrator")

        # Two LLMs: function calling and chat
        self.function_calling_model = get_function_calling_llm()
        self.chat_model = get_chat_llm()

        self.coordinator = MultiAgentCoordinator()

        # Register agents
        if agents:
            self._register_agents(agents)
        else:
            self._setup_default_agents()

        self.workflow = self._build_workflow()
        self.logger.info("Refactored MultiStageOrchestrator initialized successfully")

    def _register_agents(self, agents):
        """Register provided agent instances"""
        self.logger.info(f"Registering {len(agents)} provided agents")
        for agent in agents:
            try:
                self.coordinator.register_agent(agent)
                self.logger.info(f"✅ Registered agent: {agent.name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to register agent {getattr(agent, 'name', 'unknown')}: {e}")

    def _setup_default_agents(self):
        """Setup default agents"""
        self.logger.info("Setting up default agents")
        try:
            from agents.vision_agent import VisionAgent
            self.coordinator.register_agent(VisionAgent())
            self.logger.info("✅ Registered default VisionAgent")
        except Exception as e:
            self.logger.error(f"❌ Failed to register VisionAgent: {e}")

        try:
            from agents.transcription_agent import TranscriptionAgent
            self.coordinator.register_agent(TranscriptionAgent())
            self.logger.info("✅ Registered default TranscriptionAgent")
        except Exception as e:
            self.logger.error(f"❌ Failed to register TranscriptionAgent: {e}")

    # ========== ROUTING FUNCTIONS ==========

    def _route_after_selection(self, state: OrchestratorState) -> Literal["plan_tools", "retry_selection", "clarify"]:
        """Route after agent selection"""
        # Check if we have selected agents
        if state.selected_agents:
            self.logger.debug(f"✅ Agents selected: {state.selected_agents} → plan_tools")
            return "plan_tools"

        # No agents selected - retry or clarify
        if state.retry_count < state.max_retries:
            self.logger.debug(f"⚠️ No agents selected, retry {state.retry_count + 1}/{state.max_retries} → retry_selection")
            return "retry_selection"

        self.logger.debug("❌ No agents selected after max retries → clarify")
        return "clarify"

    def _route_after_planning(self, state: OrchestratorState) -> Literal["execute_agent", "retry_planning", "clarify"]:
        """Route after tool planning"""
        # Check if we have valid execution plans
        has_plans = bool(state.execution_plans) and all(
            len(tools) > 0 for tools in state.execution_plans.values()
        )

        if has_plans:
            self.logger.debug(f"✅ Tools planned: {state.execution_plans} → execute_agent")
            return "execute_agent"

        # No valid plans - retry or clarify
        if state.retry_count < state.max_retries:
            self.logger.debug(f"⚠️ No valid plans, retry {state.retry_count + 1}/{state.max_retries} → retry_planning")
            return "retry_planning"

        self.logger.debug("❌ No valid plans after max retries → clarify")
        return "clarify"

    def _route_after_execution(self, state: OrchestratorState) -> Literal["execute_agent", "generate_response"]:
        """Route after executing an agent - continue loop or finish"""
        if state.current_agent_index < len(state.execution_plans):
            self.logger.debug(f"🔄 More agents to execute ({state.current_agent_index}/{len(state.execution_plans)}) → execute_agent")
            return "execute_agent"

        self.logger.debug("✅ All agents executed → generate_response")
        return "generate_response"

    # ========== NODES ==========

    def _select_agents_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Select which agents to use for this task"""
        self.logger.info("📍 Node: select_agents")

        available_agents_dict = self.coordinator.get_available_agents()
        task_description = state.task_request.task.get_task_description()
        selected_agents: List[str] = []
        planner_llm_calls = state.planner_llm_calls

        # Try intent-based routing first
        if Config.USE_INTENT_ROUTING:
            from routing.intent_classifier import get_intent_classifier
            classifier = get_intent_classifier()
            intent_matches = classifier.classify(task_description)
            min_conf = Config.INTENT_CONFIDENCE_THRESHOLD

            if intent_matches and intent_matches[0][1] >= min_conf:
                selected_agents = [agent_name for agent_name, score in intent_matches[:2]]
                self.logger.info(f"🎯 Intent-based selection: {selected_agents}")

        # Fallback to LLM-based selection
        if not selected_agents:
            self.logger.info("🤖 Using LLM-based agent selection")

            available_agents = "\n".join([
                f"- {name}: {', '.join(capabilities)}"
                for name, capabilities in available_agents_dict.items()
            ])

            prompt_template = OrchestratorPrompts.AGENT_SELECTOR
            formatted_prompt = prompt_template.format(
                available_agents=list(available_agents_dict.keys()),
                agent_capabilities=available_agents,
                user_request=task_description
            )

            response = self.function_calling_model.invoke([HumanMessage(content=formatted_prompt)])
            planner_llm_calls += 1

            try:
                json_match = re.search(r'\[.*?\]', response.content)
                if json_match:
                    selected_agents = json.loads(json_match.group())
                else:
                    selected_agents = json.loads(response.content)
            except:
                self.logger.warning("Failed to parse LLM response, no agents selected")
                selected_agents = []

        return {
            "selected_agents": selected_agents,
            "planner_llm_calls": planner_llm_calls,
            "retry_count": 0,  # Reset retry for next stage
            "messages": state.messages + [
                AIMessage(content=f"Selected agents: {selected_agents}" if selected_agents else "No agents selected")
            ]
        }

    def _retry_selection_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Retry agent selection with modified prompt"""
        self.logger.info(f"🔄 Node: retry_selection (attempt {state.retry_count + 1})")

        # For retry, we could use a more lenient approach or different prompt
        # For now, just increment retry count and try again
        return {
            "retry_count": state.retry_count + 1,
            "retry_stage": "selection"
        }

    def _plan_tools_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Plan which tools each agent should use"""
        self.logger.info("📍 Node: plan_tools")

        execution_plans = {}
        planner_llm_calls = state.planner_llm_calls

        for agent_name in state.selected_agents:
            # Find agent
            agent = None
            for registered_agent in self.coordinator.agents.values():
                if registered_agent.name == agent_name:
                    agent = registered_agent
                    break

            if not agent:
                self.logger.warning(f"⚠️ Agent {agent_name} not found")
                continue

            # Get tools
            tools = agent.get_tools()
            tool_names = [tool.name for tool in tools]
            tool_descriptions = {tool.name: tool.description for tool in tools}

            # Ask LLM to plan
            prompt_template = OrchestratorPrompts.TOOL_PLANNER
            formatted_prompt = prompt_template.format(
                agent_name=agent_name,
                tool_names=tool_names,
                tool_descriptions=tool_descriptions,
                user_request=state.task_request.task.get_task_description(),
                agent_role=f"Handles {', '.join(agent.capabilities)}"
            )

            response = self.function_calling_model.invoke([HumanMessage(content=formatted_prompt)])
            planner_llm_calls += 1

            try:
                json_match = re.search(r'\[.*?\]', response.content)
                if json_match:
                    tools_to_use = json.loads(json_match.group())
                else:
                    tools_to_use = json.loads(response.content)
            except:
                # Fallback: use first tool
                tools_to_use = [tool_names[0]] if tool_names else []

            execution_plans[agent_name] = tools_to_use
            self.logger.info(f"   {agent_name}: {tools_to_use}")

        return {
            "execution_plans": execution_plans,
            "planner_llm_calls": planner_llm_calls,
            "retry_count": 0,  # Reset for next stage
            "messages": state.messages + [
                AIMessage(content=f"Planned tools: {execution_plans}")
            ]
        }

    def _retry_planning_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Retry tool planning"""
        self.logger.info(f"🔄 Node: retry_planning (attempt {state.retry_count + 1})")

        return {
            "retry_count": state.retry_count + 1,
            "retry_stage": "planning"
        }

    def _execute_agent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Execute current agent with planned tools"""
        agent_names = list(state.execution_plans.keys())
        current_agent_name = agent_names[state.current_agent_index]
        planned_tools = state.execution_plans[current_agent_name]

        self.logger.info(f"📍 Node: execute_agent [{state.current_agent_index + 1}/{len(agent_names)}] - {current_agent_name}")
        self.logger.info(f"   Tools: {planned_tools}")

        # Execute through coordinator
        result = self.coordinator.process_task_request(
            state.task_request,
            agent_name=current_agent_name,
            planned_tools=planned_tools
        )
        result_dict = result.dict()

        # Store result
        agent_results = state.agent_results.copy()
        agent_results[current_agent_name] = result_dict

        return {
            "agent_results": agent_results,
            "current_agent_index": state.current_agent_index + 1,
            "agent_llm_calls": state.agent_llm_calls + result_dict["llm_calls"],
            "messages": state.messages + [
                AIMessage(content=f"Executed {current_agent_name}")
            ]
        }

    def _generate_response_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Generate natural language response from agent results"""
        self.logger.info("📍 Node: generate_response")

        user_request = state.task_request.task.get_task_description()
        agent_results = state.agent_results

        prompt = f"""
The user asked: "{user_request}"

I have completed the analysis with the following results:
{json.dumps(agent_results, indent=2)}

Please provide a helpful, conversational response that:
1. Directly answers their question
2. Summarizes the key findings in plain language
3. Is friendly and easy to understand
4. Focuses on what the user actually cares about

Response:
"""

        response = self.chat_model.invoke([HumanMessage(content=prompt)])

        # Final polish
        polish_prompt = f"""
Original user request: "{user_request}"

Generated response: "{response.content}"

Please polish this response to make it:
1. Well-formatted and easy to read
2. Professional yet friendly
3. Complete and helpful
4. Properly structured with clear sections if needed

Final response:
"""

        final_response = self.chat_model.invoke([HumanMessage(content=polish_prompt)])

        return {
            "chat_response": response.content,
            "final_result": final_response.content,
            "chat_llm_calls": state.chat_llm_calls + 2,
            "status": "COMPLETE",
            "messages": state.messages + [
                AIMessage(content=f"Generated response")
            ]
        }

    def _clarify_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle clarification request when we can't proceed"""
        self.logger.info("📍 Node: clarify")

        # Build clarification message
        capabilities = AgentCapabilityRegistry.get_all_capabilities()
        capability_summary = "\n".join([
            f"- {name.replace('_', ' ').title()}: {', '.join(cap.capabilities[:3])}"
            for name, cap in capabilities.items()
        ])

        user_request = state.task_request.task.get_task_description()
        clarification_message = OrchestratorPrompts.format_clarification_message(
            user_request,
            capability_summary
        )

        return {
            "clarification_message": clarification_message,
            "final_result": clarification_message,
            "status": "NEED_CLARIFICATION",
            "messages": state.messages + [
                AIMessage(content="Requesting clarification from user")
            ]
        }

    # ========== WORKFLOW BUILDER ==========

    def _build_workflow(self) -> StateGraph:
        """Build workflow with conditional edges and status-based routing"""
        workflow = StateGraph(OrchestratorState)

        # Add nodes
        workflow.add_node("select_agents", self._select_agents_node)
        workflow.add_node("retry_selection", self._retry_selection_node)
        workflow.add_node("plan_tools", self._plan_tools_node)
        workflow.add_node("retry_planning", self._retry_planning_node)
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("clarify", self._clarify_node)

        # Start
        workflow.add_edge(START, "select_agents")

        # Conditional edges with routing functions
        workflow.add_conditional_edges(
            "select_agents",
            self._route_after_selection,
            {
                "plan_tools": "plan_tools",
                "retry_selection": "retry_selection",
                "clarify": "clarify"
            }
        )

        workflow.add_conditional_edges(
            "retry_selection",
            self._route_after_selection,  # Same routing logic
            {
                "plan_tools": "plan_tools",
                "retry_selection": "retry_selection",  # Can retry multiple times
                "clarify": "clarify"
            }
        )

        workflow.add_conditional_edges(
            "plan_tools",
            self._route_after_planning,
            {
                "execute_agent": "execute_agent",
                "retry_planning": "retry_planning",
                "clarify": "clarify"
            }
        )

        workflow.add_conditional_edges(
            "retry_planning",
            self._route_after_planning,  # Same routing logic
            {
                "execute_agent": "execute_agent",
                "retry_planning": "retry_planning",  # Can retry multiple times
                "clarify": "clarify"
            }
        )

        workflow.add_conditional_edges(
            "execute_agent",
            self._route_after_execution,
            {
                "execute_agent": "execute_agent",  # Loop for next agent
                "generate_response": "generate_response"
            }
        )

        # End nodes
        workflow.add_edge("generate_response", END)
        workflow.add_edge("clarify", END)

        return workflow.compile()

    # ========== MAIN ENTRY POINT ==========

    def process_task(self, task_request: TaskRequest, file_path: str = None) -> Dict[str, Any]:
        """Main entry point for processing a task"""

        # Load video into context if it's a video task
        if hasattr(task_request.task, 'file_path'):
            from context.video_context import get_video_context
            video_context = get_video_context()
            video_context.set_current_video(task_request.task.file_path)
            self.logger.info(f"📹 Loaded video: {task_request.task.file_path}")

        # Create initial state using Pydantic
        initial_state = OrchestratorState(
            task_request=task_request,
            messages=[HumanMessage(content=task_request.task.description)],
            max_retries=Config.MAX_RETRIES if hasattr(Config, 'MAX_RETRIES') else 2
        )

        self.logger.info("="*80)
        self.logger.info("🚀 Starting Orchestration")
        self.logger.info(f"📋 Task: {task_request.task.get_task_description()}")
        self.logger.info("="*80)

        # Run the workflow
        result = self.workflow.invoke(initial_state.dict())

        # Convert back to Pydantic for easy access
        final_state = OrchestratorState(**result)

        self.logger.info("="*80)
        self.logger.info("✅ Orchestration Complete")
        self.logger.info(f"📊 Status: {final_state.status}")
        self.logger.info(f"🤖 Agents: {final_state.selected_agents}")
        self.logger.info(f"🔢 LLM Calls: {final_state.planner_llm_calls + final_state.agent_llm_calls + final_state.chat_llm_calls}")
        self.logger.info("="*80)

        total_llm_calls = (
            final_state.planner_llm_calls
            + final_state.agent_llm_calls
            + final_state.chat_llm_calls
        )

        return {
            "success": final_state.status == "COMPLETE",
            "status": final_state.status,
            "task_request": final_state.task_request,
            "selected_agents": final_state.selected_agents,
            "execution_plans": final_state.execution_plans,
            "agent_results": final_state.agent_results,
            "final_result": final_state.final_result,
            "planner_llm_calls": final_state.planner_llm_calls,
            "agent_llm_calls": final_state.agent_llm_calls,
            "chat_llm_calls": final_state.chat_llm_calls,
            "total_llm_calls": total_llm_calls,
            "retry_info": {
                "retry_count": final_state.retry_count,
                "retry_stage": final_state.retry_stage,
                "max_retries": final_state.max_retries
            }
        }
