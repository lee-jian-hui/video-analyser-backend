from typing import Dict, Any, List, Literal, Union
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from graph import MessagesState
from llm import get_model
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

    """TypedDict for LangGraph state management"""
    messages: List[Any]
    llm_calls: int
    task_request: TaskRequest
    selected_agents: List[str]
    execution_plans: Dict[str, List[str]]
    agent_results: Dict[str, Any]
    current_agent_index: int
    final_result: str


class MultiStageOrchestrator:
    """Multi-stage LLM orchestration using LangGraph"""

    def __init__(self, agents=None):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MultiStageOrchestrator")
        self.model = get_model(Config.GEMINI_API_KEY)
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
        """Build the multi-stage orchestration workflow (stategraph)"""
        workflow = StateGraph(OrchestratorState)

        # Add decision nodes
        workflow.add_node("agent_selector", self._agent_selector_node)
        workflow.add_node("tool_planner", self._tool_planner_node)
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("aggregate_results", self._aggregate_results_node)

        # Add edges
        workflow.add_edge(START, "agent_selector")
        workflow.add_edge("agent_selector", "tool_planner")
        workflow.add_edge("tool_planner", "execute_agent")

        # Conditional edge: continue execution or aggregate
        workflow.add_conditional_edges(
            "execute_agent",
            self._execution_router,
            {
                "continue_execution": "execute_agent",
                "aggregate": "aggregate_results"
            }
        )
        workflow.add_edge("aggregate_results", END)

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

        response = self.model.invoke([HumanMessage(content=formatted_prompt)])

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

            response = self.model.invoke([HumanMessage(content=formatted_prompt)])

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

    def _execution_router(self, state: OrchestratorState) -> Literal["continue_execution", "aggregate"]:
        """Route between continuing execution or aggregating results"""
        self.logger.debug(f"Execution router - current_agent_index: {state['current_agent_index']}, execution_plans length: {len(state['execution_plans'])}")
        if state["current_agent_index"] < len(state["execution_plans"]):
            self.logger.debug("Routing to: continue_execution")
            return "continue_execution"
        self.logger.debug("Routing to: aggregate")
        return "aggregate"

    def _aggregate_results_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Final stage: Aggregate all agent results"""
        # Use template
        prompt_template = OrchestratorPrompts.RESULT_AGGREGATOR
        formatted_prompt = prompt_template.format(
            original_task=state['task_request'].task.get_task_description(),
            agent_results=json.dumps(state['agent_results'], indent=2)
        )

        response = self.model.invoke([HumanMessage(content=formatted_prompt)])

        return {
            "final_result": response.content,
            "messages": state["messages"] + [
                AIMessage(content=f"Final Result: {response.content}")
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
            "selected_agents": [],
            "execution_plans": {},
            "agent_results": {},
            "current_agent_index": 0,
            "final_result": ""
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