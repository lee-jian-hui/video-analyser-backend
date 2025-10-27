import logging
from typing import Dict, Any, List, Literal, Union
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from graph import MessagesState
from llm import get_function_calling_llm, get_chat_llm
from multi_agent_coordinator import MultiAgentCoordinator
from models.orchestrator_models import AgentResult, OrchestrationResult
from models.task_models import TaskRequest, VideoTask, ImageTask, TextTask
from models.agent_capabilities import AgentCapabilityRegistry
from templates.orchestrator_prompts import OrchestratorPrompts, PromptExamples
from utils.logger import get_logger
from configs import Config
from typing_extensions import TypedDict
import time
import os
import json
import re

class OrchestratorState(TypedDict):
    """Extended state for function calling vs chat separation"""
    messages: List[Any]
    llm_calls: int
    task_request: TaskRequest
    planner_llm_calls: int
    agent_llm_calls: int
    chat_llm_calls: int
    clarification_active: bool
    clarification_message: str
    
    # FUNCTION CALLING RESULTS
    selected_agents: List[str]
    execution_plans: Dict[str, List[str]]
    agent_results: Dict[str, Any]
    current_agent_index: int
    
    # CHAT RESULTS  
    chat_response: str          # NEW: Natural language response
    final_result: str           # NEW: Final formatted output
    
    # METADATA
    function_calling_steps: int  # NEW: Track function calling usage
    chat_steps: int             # NEW: Track chat usage
    
    # DEADLINES
    call_deadline_ts: float     # NEW: Absolute deadline for the whole call (epoch seconds)
    
    # TOOLS NEEDED GATE
    tools_needed: bool          # NEW: Whether to run tools for this request
    tools_reason: str           # NEW: Short rationale from the model


class MultiStageOrchestrator:
    """Multi-stage LLM orchestration using LangGraph"""

    def __init__(self, agents=None):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MultiStageOrchestrator")
        
        # TWO SEPARATE LLMs
        self.function_calling_model = get_function_calling_llm()  # For structured decisions
        self.chat_model = get_chat_llm()                         # For natural responses
        
        self.coordinator = MultiAgentCoordinator()

        # Register provided agents or use default setup
        if agents:
            self._register_agents(agents)
        else:
            self._setup_default_agents()

        self.workflow = self._build_workflow()
        self.logger.info("MultiStageOrchestrator initialized successfully")
    
    def _log_state(self, label: str, state: OrchestratorState):
        """Debug helper to print orchestrator state snapshot."""
        if self.logger.isEnabledFor(logging.DEBUG):
            snapshot = {
                "label": label,
                "selected_agents": state.get("selected_agents"),
                "execution_plans": state.get("execution_plans"),
                "agent_results_keys": list(state.get("agent_results", {}).keys()),
                "planner_llm_calls": state.get("planner_llm_calls"),
                "agent_llm_calls": state.get("agent_llm_calls"),
                "chat_llm_calls": state.get("chat_llm_calls"),
                "current_agent_index": state.get("current_agent_index"),
                "clarification_active": state.get("clarification_active"),
            }
            self.logger.debug(f"Orchestrator state: {snapshot}")
    
    def _log_next_state(self, label: str, state: OrchestratorState, update: Dict[str, Any]):
        merged = state.copy()
        merged.update(update)
        self._log_state(label, merged)  # type: ignore[arg-type]

    def _register_agents(self, agents):
        """Register provided agent instances"""
        self.logger.info(f"Registering {len(agents)} provided agents")

        for agent in agents:
            try:
                self.coordinator.register_agent(agent)
                self.logger.info(f"Successfully registered agent: {agent.name}")
            except Exception as e:
                self.logger.error(f"Failed to register agent {getattr(agent, 'name', 'unknown')}: {e}")

    def _setup_default_agents(self):
        """Setup default agents (fallback when no agents provided)"""
        self.logger.info("Setting up default agents")

        try:
            from agents.vision_agent import VisionAgent
            self.coordinator.register_agent(VisionAgent())
            self.logger.info("Successfully registered default VisionAgent")
        except Exception as e:
            self.logger.error(f"Failed to register default VisionAgent: {e}")

        try:
            from agents.transcription_agent import TranscriptionAgent
            self.coordinator.register_agent(TranscriptionAgent())
            self.logger.info("Successfully registered default TranscriptionAgent")
        except Exception as e:
            self.logger.error(f"Failed to register default TranscriptionAgent: {e}")

        try:
            from agents.reclarify_agent import ReclarifyAgent
            self.coordinator.register_agent(ReclarifyAgent())
            self.logger.info("Successfully registered default ReclarifyAgent")
        except Exception as e:
            self.logger.error(f"Failed to register default ReclarifyAgent: {e}")

    def _build_workflow(self) -> StateGraph:
        """Build workflow using Command pattern for dynamic routing"""
        workflow = StateGraph(OrchestratorState)

        # Add all nodes
        workflow.add_node("agent_selector", self._agent_selector_node)
        workflow.add_node("tools_needed_gate", self._tools_needed_gate_node)
        workflow.add_node("tool_planner", self._tool_planner_node)
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("clarification_request", self._clarification_node)
        workflow.add_node("response_generator", self._response_generator_node)
        workflow.add_node("final_formatter", self._final_formatter_node)

        # Start edge only - Command pattern in each node handles all other routing
        # This prevents conflicts where multiple edges try to execute simultaneously
        workflow.add_edge(START, "agent_selector")

        # All routing is handled by Command.goto() in each node:
        # - agent_selector: routes to "tools_needed_gate" OR "clarification_request"
        # - tool_planner: routes to "execute_agent" OR "clarification_request"
        # - execute_agent: routes to "execute_agent" (loop) OR "response_generator"
        # - clarification_request: routes to "final_formatter"
        # - response_generator: routes to "final_formatter"
        # - final_formatter: is END node (no Command return)

        return workflow.compile()

    def _build_capability_summary(self) -> str:
        capabilities = AgentCapabilityRegistry.get_all_capabilities()
        if not capabilities:
            return "- No specialized agents are currently registered."

        lines: List[str] = []
        for agent_name, capability in capabilities.items():
            summary = ", ".join(capability.capabilities[:3]) or "No documented capabilities"
            display_name = agent_name.replace("_", " ").title()
            lines.append(f"- {display_name}: {summary}")

        return "\n".join(lines)

    def _build_clarification_message(self, user_request: str) -> str:
        summary = self._build_capability_summary()
        return OrchestratorPrompts.format_clarification_message(user_request, summary)

    def _agent_selector_node(self, state: OrchestratorState) -> Command[Literal["tool_planner"]]:
        """FUNCTION CALLING: Select which agents to use"""
        available_agents_dict = self.coordinator.get_available_agents()
        task_description = state['task_request'].task.get_task_description()

        selected_agents: List[str] = []
        planner_llm_calls = state.get("planner_llm_calls", 0)

        if Config.USE_INTENT_ROUTING:
            from routing.intent_classifier import get_intent_classifier
            classifier = get_intent_classifier()
            intent_matches = classifier.classify(task_description)
            min_conf = Config.INTENT_CONFIDENCE_THRESHOLD

            if intent_matches and intent_matches[0][1] >= min_conf:
                selected_agents = [agent_name for agent_name, score in intent_matches[:2]]
                self.logger.info(f"🎯 Intent-based agent selection: '{task_description}'")
                self.logger.info(f"   → Selected agents: {selected_agents} (scores: {[f'{s:.2f}' for _, s in intent_matches[:2]]})")

        if not selected_agents:
            self.logger.info(f"⚠️  Using LLM-based selection for: '{task_description}'")

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
                selected_agents = ["reclarify_agent"]

            self.logger.info(f"   → LLM selected agents: {selected_agents}")

        clarification_needed = not selected_agents
        clarification_message = self._build_clarification_message(task_description) if clarification_needed else ""

        update = {
            "selected_agents": selected_agents,
            "function_calling_steps": state.get('function_calling_steps', 0) + 1,
            "planner_llm_calls": planner_llm_calls,
            "clarification_active": clarification_needed,
            "clarification_message": clarification_message,
            "messages": state["messages"] + [
                AIMessage(content=f"Selected agents: {selected_agents}" if selected_agents else "No agent matched; requesting clarification.")
            ]
        }
        self._log_next_state("after_agent_selector", state, update)
        next_node = "tools_needed_gate" if selected_agents else "clarification_request"
        return Command(update=update, goto=next_node)

    def _tools_needed_gate_node(self, state: OrchestratorState) -> Command[Literal["tool_planner", "response_generator"]]:
        """Decide whether to run tools or respond conversationally"""
        user_request = state['task_request'].task.get_task_description()

        prompt_template = OrchestratorPrompts.TOOLS_NEEDED_GATE
        formatted_prompt = prompt_template.format(user_request=user_request)

        tools_needed = False
        reason = "Defaulting to conversation; tool gating parse failed."

        try:
            response = self.function_calling_model.invoke([HumanMessage(content=formatted_prompt)])
            import json as _json
            # Try to extract JSON object
            m = re.search(r'\{.*\}', response.content, re.DOTALL)
            raw = m.group(0) if m else response.content
            data = _json.loads(raw)
            tools_needed = bool(data.get("should_use_tools", False))
            reason = str(data.get("reason", "")) or reason
        except Exception as e:
            self.logger.info(f"Tools-needed gate parse error; defaulting to chat. Error: {e}")

        # If tools not needed, switch to reclarify agent so we still use a tool path
        next_selected = state["selected_agents"]
        if not tools_needed:
            next_selected = ["reclarify_agent"]

        update = {
            "tools_needed": tools_needed,
            "tools_reason": reason,
            "selected_agents": next_selected,
            "messages": state["messages"] + [
                AIMessage(content=f"Tools-needed decision: {tools_needed} ({reason}) → agent(s): {next_selected}")
            ],
            "function_calling_steps": state.get('function_calling_steps', 0) + 1,
        }
        self._log_next_state("after_tools_needed_gate", state, update)

        next_node = "tool_planner"
        return Command(update=update, goto=next_node)

    def _tool_planner_node(self, state: OrchestratorState) -> Command[Literal["execute_agent"]]:
        """FUNCTION CALLING: Plan tools for each selected agent"""
        execution_plans = {}

        planner_llm_calls = state.get("planner_llm_calls", 0)

        for agent_name in state["selected_agents"]:
            # Get agent's available tools
            agent = None
            for registered_agent in self.coordinator.agents.values():
                if registered_agent.name == agent_name:
                    agent = registered_agent
                    break

            if not agent:
                continue

            tools = agent.get_tools()
            tool_names = [tool.name for tool in tools]
            tool_descriptions = {tool.name: tool.description for tool in tools}

            # Debug: Log available tools
            self.logger.debug(f"Available tools for {agent_name}: {tool_names}")
            self.logger.debug(f"Tool descriptions: {tool_descriptions}")

            # Use template
            prompt_template = OrchestratorPrompts.TOOL_PLANNER
            formatted_prompt = prompt_template.format(
                agent_name=agent_name,
                tool_names=tool_names,
                tool_descriptions=tool_descriptions,
                user_request=state['task_request'].task.get_task_description(),
                agent_role=f"Handles {', '.join(agent.capabilities)}"
            )

            self.logger.debug(f"Tool planner prompt: {formatted_prompt}")

            response = self.function_calling_model.invoke([HumanMessage(content=formatted_prompt)])
            planner_llm_calls += 1

            try:
                # Extract JSON from response
                json_match = re.search(r'\[.*?\]', response.content)
                if json_match:
                    tools = json.loads(json_match.group())
                else:
                    tools = json.loads(response.content)
            except:
                # Fallback: use first available tool
                tools = [tool_names[0]] if tool_names else []

            execution_plans[agent_name] = tools

        has_valid_plan = bool(execution_plans) and all(len(tools) > 0 for tools in execution_plans.values())
        clarification_message = ""

        if not has_valid_plan:
            clarification_message = self._build_clarification_message(
                state['task_request'].task.get_task_description()
            )

        update = {
            "execution_plans": execution_plans,
            "function_calling_steps": state.get('function_calling_steps', 0) + 1,
            "planner_llm_calls": planner_llm_calls,
            "clarification_active": not has_valid_plan,
            "clarification_message": clarification_message,
            "messages": state["messages"] + [
                AIMessage(content="Execution plans: {}".format(execution_plans) if has_valid_plan else "No valid tool plan generated; requesting clarification from the user.")
            ]
        }
        self._log_next_state("after_tool_planner", state, update)
        next_node = "execute_agent" if has_valid_plan else "clarification_request"
        return Command(update=update, goto=next_node)

    def _execute_agent_node(self, state: OrchestratorState) -> Command[Literal["execute_agent", "response_generator"]]:
        """FUNCTION CALLING: Execute current agent with planned tools"""
        self.logger.debug(f"Executing agent node, current_agent_index: {state['current_agent_index']}")
        agent_names = list(state["execution_plans"].keys())
        self.logger.debug(f"Agent names: {agent_names}")

        if state["current_agent_index"] >= len(agent_names):
            self.logger.debug("All agents executed, moving to response generation")
            return Command(update={}, goto="response_generator")

        current_agent_name = agent_names[state["current_agent_index"]]
        planned_tools = state["execution_plans"][current_agent_name]
        self.logger.debug(f"Executing {current_agent_name} with tools: {planned_tools}")

        # Compute per-agent time budget based on remaining call budget
        from configs import Config
        now = time.time()
        remaining_call_s = max(0.0, state["call_deadline_ts"] - now)
        safety = max(0.0, Config.SCHEDULER_SAFETY_MARGIN_S)

        if remaining_call_s <= safety:
            self.logger.info("⏱️  Call deadline nearly reached; skipping remaining agents")
            return Command(update={}, goto="response_generator")

        agent_default = Config.PER_AGENT_DEFAULT_BUDGET_S
        per_agent_map = getattr(Config, "AGENT_BUDGETS_S", {}) or {}
        configured_budget = per_agent_map.get(current_agent_name, agent_default)
        agent_budget_s = max(0.0, min(configured_budget, max(0.0, remaining_call_s - safety)))
        self.logger.debug(f"Computed budget for {current_agent_name}: {agent_budget_s:.2f}s (remaining call {remaining_call_s:.2f}s)")

        # Execute through coordinator with the task request and budget
        result = self.coordinator.process_task_request(
            state['task_request'],
            agent_name=current_agent_name,
            planned_tools=planned_tools,
            time_budget_s=agent_budget_s
        )
        result_dict = result.dict()
        self.logger.debug(f"Agent execution result: {result_dict}")

        # Store result
        agent_results = state["agent_results"].copy()
        agent_results[current_agent_name] = result_dict
        agent_llm_calls = state.get("agent_llm_calls", 0) + result_dict["llm_calls"]
        
        new_agent_index = state["current_agent_index"] + 1
        
        # Determine next step: continue with more agents or go to response generation
        if new_agent_index < len(agent_names):
            next_step = "execute_agent"
        else:
            next_step = "response_generator"

        update = {
            "agent_results": agent_results,
            "current_agent_index": new_agent_index,
            "agent_llm_calls": agent_llm_calls,
            "messages": state["messages"] + [
                AIMessage(content=f"Executed {current_agent_name}: {result_dict.get('messages', [])}")
            ]
        }
        self._log_next_state("after_execute_agent", state, update)
        next_node = "execute_agent" if new_agent_index < len(agent_names) else "response_generator"
        return Command(update=update, goto=next_node)


    def _clarification_node(self, state: OrchestratorState) -> Command[Literal["final_formatter"]]:
        """Fallback when no agents/tools were selected."""
        message = state.get("clarification_message") or self._build_clarification_message(
            state['task_request'].task.get_task_description()
        )
        update = {
            "clarification_active": True,
            "chat_response": message,
            "messages": state["messages"] + [AIMessage(content=message)]
        }
        self._log_next_state("after_clarification", state, update)
        return Command(update=update, goto="final_formatter")


    def _response_generator_node(self, state: OrchestratorState) -> Command[Literal["final_formatter"]]:
        """CHAT: Generate natural language response from agent results"""
        agent_results = state.get('agent_results', {})
        user_request = state['task_request'].task.get_task_description()
        
        # Create a conversational prompt for the chat model
        prompt = f"""
        The user asked: "{user_request}"
        
        I have completed the analysis with the following results:
        {json.dumps(agent_results, indent=2)}
        
        Please provide a helpful, conversational response to the user that:
        1. Directly answers their question
        2. Summarizes the key findings in plain language
        3. Is friendly and easy to understand
        4. Focuses on what the user actually cares about
        
        Response:
        """
        
        # Use chat LLM for natural conversation
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        chat_llm_calls = state.get("chat_llm_calls", 0) + 1

        update = {
            "chat_response": response.content,
            "chat_llm_calls": chat_llm_calls,
            "chat_steps": state.get('chat_steps', 0) + 1,
            "messages": state["messages"] + [
                AIMessage(content=f"Generated response: {response.content}")
            ]
        }
        self._log_next_state("after_response_generator", state, update)
        return Command(update=update, goto="final_formatter")

    def _final_formatter_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """CHAT: Final formatting and polish of the response - END node"""
        chat_response = state.get('chat_response', '')
        user_request = state['task_request'].task.get_task_description()
        
        # Polish the response with the chat model
        prompt = f"""
        Original user request: "{user_request}"
        
        Generated response: "{chat_response}"
        
        Please polish this response to make it:
        1. Well-formatted and easy to read
        2. Professional yet friendly
        3. Complete and helpful
        4. Properly structured with clear sections if needed
        
        Final response:
        """
        
        if state.get("clarification_active"):
            return {
                "final_result": chat_response,
                "chat_llm_calls": state.get("chat_llm_calls", 0),
                "chat_steps": state.get('chat_steps', 0),
                "messages": state["messages"]
            }

        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        chat_llm_calls = state.get("chat_llm_calls", 0) + 1

        return {
            "final_result": response.content,
            "chat_llm_calls": chat_llm_calls,
            "chat_steps": state.get('chat_steps', 0) + 1,
            "messages": state["messages"] + [
                AIMessage(content=f"Final formatted result: {response.content}")
            ]
        }

    def process_task(self, task_request: TaskRequest, file_path: str = None) -> Dict[str, Any]:
        """Main entry point for processing a task"""
        # Handle backward compatibility - convert string to TaskRequest
        # if isinstance(task_request, str):
        #     from models.task_models import VideoTask
        #     task_request = TaskRequest(
        #         task=VideoTask(
        #             description=task_request,
        #             file_path=file_path or "./sample.mp4",
        #             task_type="object_detection"
        #         )
        #     )

        # Load video into context if it's a video task
        if hasattr(task_request.task, 'file_path'):
            from context.video_context import get_video_context
            video_context = get_video_context()
            video_context.set_current_video(task_request.task.file_path)
            self.logger.info(f"📹 Loaded video into context: {task_request.task.file_path}")

        from configs import Config
        call_deadline_ts = time.time() + max(0.0, Config.ORCHESTRATION_TOTAL_TIMEOUT_S)

        initial_state = {
            "messages": [HumanMessage(content=task_request.task.description)],
            "llm_calls": 0,
            "task_request": task_request,
            "planner_llm_calls": 0,
            "agent_llm_calls": 0,
            "chat_llm_calls": 0,
            "clarification_active": False,
            "clarification_message": "",
            "call_deadline_ts": call_deadline_ts,
            "tools_needed": True,
            "tools_reason": "",

            # FUNCTION CALLING RESULTS
            "selected_agents": [],
            "execution_plans": {},
            "agent_results": {},
            "current_agent_index": 0,

            # CHAT RESULTS
            "chat_response": "",
            "final_result": "",

            # METADATA
            "function_calling_steps": 0,
            "chat_steps": 0
        }

        # Run the workflow
        result = self.workflow.invoke(initial_state)
        self._log_state("final", result)  # type: ignore[arg-type]

        total_llm_calls = (
            result["planner_llm_calls"]
            + result["agent_llm_calls"]
            + result["chat_llm_calls"]
        )

        return {
            "success": True,
            "task_request": result["task_request"],
            "selected_agents": result["selected_agents"],
            "execution_plans": result["execution_plans"],
            "agent_results": result["agent_results"],
            "final_result": result["final_result"],
            "planner_llm_calls": result["planner_llm_calls"],
            "agent_llm_calls": result["agent_llm_calls"],
            "chat_llm_calls": result["chat_llm_calls"],
            "total_llm_calls": total_llm_calls
        }
