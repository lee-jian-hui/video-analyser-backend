from typing import Dict, Any, List, Literal, Union
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from graph import MessagesState
from llm import get_function_calling_llm, get_chat_llm
from multi_agent_coordinator import MultiAgentCoordinator
from models.orchestrator_models import AgentResult, OrchestrationResult
from models.task_models import TaskRequest, VideoTask, ImageTask, TextTask
from templates.orchestrator_prompts import OrchestratorPrompts, PromptExamples
from utils.logger import get_logger
from configs import Config
from typing_extensions import TypedDict
import os
import json
import re

class OrchestratorState(TypedDict):
    """Extended state for function calling vs chat separation"""
    messages: List[Any]
    llm_calls: int
    task_request: TaskRequest
    
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

    def _build_workflow(self) -> StateGraph:
        """Build workflow using idiomatic Command pattern"""
        workflow = StateGraph(OrchestratorState)

        # Add all nodes - routing handled by Command pattern
        workflow.add_node("agent_selector", self._agent_selector_node)      # Structured decision
        workflow.add_node("tool_planner", self._tool_planner_node)          # Structured planning
        workflow.add_node("execute_agent", self._execute_agent_node)        # Tool execution
        workflow.add_node("response_generator", self._response_generator_node)  # Natural language
        workflow.add_node("final_formatter", self._final_formatter_node)        # User-friendly output

        # Only need start edge - Command pattern handles all routing
        workflow.add_edge(START, "agent_selector")
        workflow.add_edge("final_formatter", END)

        return workflow.compile()

    def _agent_selector_node(self, state: OrchestratorState) -> Command[Literal["tool_planner"]]:
        """FUNCTION CALLING: Select which agents to use"""
        available_agents_dict = self.coordinator.get_available_agents()
        task_description = state['task_request'].task.get_task_description()

        # NEW: Try intent-based routing first
        from routing.intent_classifier import get_intent_classifier
        classifier = get_intent_classifier()
        intent_matches = classifier.classify(task_description)

        if intent_matches:
            selected_agents = [agent_name for agent_name, score in intent_matches[:2]]  # Top 2 agents
            self.logger.info(f"🎯 Intent-based agent selection: '{task_description}'")
            self.logger.info(f"   → Selected agents: {selected_agents} (scores: {[f'{s:.2f}' for _, s in intent_matches[:2]]})")
        else:
            # Fallback to LLM-based selection
            self.logger.info(f"⚠️  No intent match, using LLM-based selection for: '{task_description}'")

            # Format available agents for template
            available_agents = "\n".join([
                f"- {name}: {', '.join(capabilities)}"
                for name, capabilities in available_agents_dict.items()
            ])

            # Use template
            prompt_template = OrchestratorPrompts.AGENT_SELECTOR
            formatted_prompt = prompt_template.format(
                available_agents=list(available_agents_dict.keys()),
                agent_capabilities=available_agents,
                user_request=task_description
            )

            response = self.function_calling_model.invoke([HumanMessage(content=formatted_prompt)])

            try:
                # Extract JSON from response
                json_match = re.search(r'\[.*?\]', response.content)
                if json_match:
                    selected_agents = json.loads(json_match.group())
                else:
                    # Fallback: try to parse the entire response
                    selected_agents = json.loads(response.content)
            except:
                # Fallback: default to vision agent
                selected_agents = ["vision_agent"]

            self.logger.info(f"   → LLM selected agents: {selected_agents}")

        return Command(
            update={
                "selected_agents": selected_agents,
                "function_calling_steps": state.get('function_calling_steps', 0) + 1,
                "messages": state["messages"] + [
                    AIMessage(content=f"Selected agents: {selected_agents}")
                ]
            },
            goto="tool_planner"
        )

    def _tool_planner_node(self, state: OrchestratorState) -> Command[Literal["execute_agent"]]:
        """FUNCTION CALLING: Plan tools for each selected agent"""
        execution_plans = {}

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

        return Command(
            update={
                "execution_plans": execution_plans,
                "function_calling_steps": state.get('function_calling_steps', 0) + 1,
                "messages": state["messages"] + [
                    AIMessage(content=f"Execution plans: {execution_plans}")
                ]
            },
            goto="execute_agent"
        )

    def _execute_agent_node(self, state: OrchestratorState) -> Command[Literal["execute_agent", "response_generator"]]:
        """FUNCTION CALLING: Execute current agent with planned tools"""
        self.logger.debug(f"Executing agent node, current_agent_index: {state['current_agent_index']}")
        agent_names = list(state["execution_plans"].keys())
        self.logger.debug(f"Agent names: {agent_names}")

        if state["current_agent_index"] >= len(agent_names):
            self.logger.debug("All agents executed, moving to response generation")
            return Command(
                update={},
                goto="response_generator"
            )

        current_agent_name = agent_names[state["current_agent_index"]]
        planned_tools = state["execution_plans"][current_agent_name]
        self.logger.debug(f"Executing {current_agent_name} with tools: {planned_tools}")

        # Execute through coordinator with the task request
        result = self.coordinator.process_task_request(
            state['task_request'],
            agent_name=current_agent_name,
            planned_tools=planned_tools
        )
        self.logger.debug(f"Agent execution result: {result}")

        # Store result
        agent_results = state["agent_results"].copy()
        agent_results[current_agent_name] = result
        
        new_agent_index = state["current_agent_index"] + 1
        
        # Determine next step: continue with more agents or go to response generation
        if new_agent_index < len(agent_names):
            next_step = "execute_agent"
        else:
            next_step = "response_generator"

        return Command(
            update={
                "agent_results": agent_results,
                "current_agent_index": new_agent_index,
                "messages": state["messages"] + [
                    AIMessage(content=f"Executed {current_agent_name}: {result.get('messages', [])}")
                ]
            },
            goto=next_step
        )


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
        
        return Command(
            update={
                "chat_response": response.content,
                "chat_steps": state.get('chat_steps', 0) + 1,
                "messages": state["messages"] + [
                    AIMessage(content=f"Generated response: {response.content}")
                ]
            },
            goto="final_formatter"
        )

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
        
        # Use chat LLM for final formatting
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        
        return {
            "final_result": response.content,
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

        initial_state = {
            "messages": [HumanMessage(content=task_request.task.description)],
            "llm_calls": 0,
            "task_request": task_request,

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

        return {
            "success": True,
            "task_request": result["task_request"],
            "selected_agents": result["selected_agents"],
            "execution_plans": result["execution_plans"],
            "agent_results": result["agent_results"],
            "final_result": result["final_result"],
            "total_llm_calls": result["llm_calls"]
        }
