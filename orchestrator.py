from typing import Dict, Any, List, Literal, Union
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
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

    def _build_workflow(self) -> StateGraph:
        """Build workflow with separate function calling and chat steps"""
        workflow = StateGraph(OrchestratorState)

        # FUNCTION CALLING STEPS (use function_calling_model)
        workflow.add_node("agent_selector", self._agent_selector_node)      # Structured decision
        workflow.add_node("tool_planner", self._tool_planner_node)          # Structured planning
        workflow.add_node("execute_agent", self._execute_agent_node)        # Tool execution
        
        # CHAT STEPS (use chat_model) 
        workflow.add_node("response_generator", self._response_generator_node)  # Natural language
        workflow.add_node("final_formatter", self._final_formatter_node)        # User-friendly output

        # EDGES
        workflow.add_edge(START, "agent_selector")
        workflow.add_edge("agent_selector", "tool_planner") 
        workflow.add_edge("tool_planner", "execute_agent")
        
        # Conditional edge: continue execution or go to chat
        workflow.add_conditional_edges(
            "execute_agent",
            self._execution_router,
            {
                "continue_execution": "execute_agent",
                "generate_response": "response_generator"  # NEW: Go to chat step
            }
        )
        workflow.add_edge("response_generator", "final_formatter")
        workflow.add_edge("final_formatter", END)

        return workflow.compile()

    def _agent_selector_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Stage 1: LLM selects which agents to use"""
        available_agents_dict = self.coordinator.get_available_agents()

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
            user_request=state['task_request'].task.get_task_description()
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

        return {
            "selected_agents": selected_agents,
            "messages": state["messages"] + [
                AIMessage(content=f"Selected agents: {selected_agents}")
            ]
        }

    def _tool_planner_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Stage 2: LLM plans tools for each selected agent"""
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

        return {
            "execution_plans": execution_plans,
            "messages": state["messages"] + [
                AIMessage(content=f"Execution plans: {execution_plans}")
            ]
        }

    def _execute_agent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Stage 3: Execute current agent with planned tools"""
        self.logger.debug(f"Executing agent node, current_agent_index: {state['current_agent_index']}")
        agent_names = list(state["execution_plans"].keys())
        self.logger.debug(f"Agent names: {agent_names}")

        if state["current_agent_index"] >= len(agent_names):
            self.logger.debug("No more agents to execute")
            return state  # No more agents to execute

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

        return {
            "agent_results": agent_results,
            "current_agent_index": state["current_agent_index"] + 1,
            "messages": state["messages"] + [
                AIMessage(content=f"Executed {current_agent_name}: {result.get('messages', [])}")
            ]
        }

    def _execution_router(self, state: OrchestratorState) -> Literal["continue_execution", "generate_response"]:
        """Route between continuing execution or aggregating results"""
        self.logger.debug(f"Execution router - current_agent_index: {state['current_agent_index']}, execution_plans length: {len(state['execution_plans'])}")
        if state["current_agent_index"] < len(state["execution_plans"]):
            self.logger.debug("Routing to: continue_execution")
            return "continue_execution"
        self.logger.debug("Routing to: generate_response")
        return "generate_response"

    def _aggregate_results_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Final stage: Aggregate all agent results"""
        # Use template
        prompt_template = OrchestratorPrompts.RESULT_AGGREGATOR
        formatted_prompt = prompt_template.format(
            original_task=state['task_request'].task.get_task_description(),
            agent_results=json.dumps(state['agent_results'], indent=2)
        )

        response = self.function_calling_model.invoke([HumanMessage(content=formatted_prompt)])

        return {
            "final_result": response.content,
            "messages": state["messages"] + [
                AIMessage(content=f"Final Result: {response.content}")
            ]
        }

    def _response_generator_node(self, state: OrchestratorState) -> Dict[str, Any]:
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
        
        return {
            "chat_response": response.content,
            "chat_steps": state.get('chat_steps', 0) + 1,
            "messages": state["messages"] + [
                AIMessage(content=f"Generated response: {response.content}")
            ]
        }

    def _final_formatter_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """CHAT: Final formatting and polish of the response"""
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

    def process_task(self, task_request: Union[TaskRequest, str], file_path: str = None) -> Dict[str, Any]:
        """Main entry point for processing a task"""
        # Handle backward compatibility - convert string to TaskRequest
        if isinstance(task_request, str):
            from models.task_models import VideoTask
            task_request = TaskRequest(
                task=VideoTask(
                    description=task_request,
                    file_path=file_path or "./sample.mp4",
                    task_type="object_detection"
                )
            )

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