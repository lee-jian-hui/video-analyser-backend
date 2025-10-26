from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class OrchestratorState(BaseModel):
    """State model for orchestration workflow"""
    messages: List[Any] = Field(default_factory=list, description="Message history")
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    original_task: str = Field(default="", description="Original user task")
    selected_agents: List[str] = Field(default_factory=list, description="Selected agents for task")
    execution_plans: Dict[str, List[str]] = Field(default_factory=dict, description="Tool execution plans per agent")
    agent_results: Dict[str, Any] = Field(default_factory=dict, description="Results from each agent")
    current_agent_index: int = Field(default=0, description="Current agent being executed")
    final_result: str = Field(default="", description="Final aggregated result")

    class Config:
        arbitrary_types_allowed = True  # Allow LangChain message types


class TaskRequest(BaseModel):
    """Model for incoming task requests"""
    task_type: str = Field(..., description="Type of task (vision, transcription, generation)")
    content: str = Field(..., description="Task content or description")
    options: Dict[str, Any] = Field(default_factory=dict, description="Optional parameters")
    execution_mode: str = Field(default="single", description="Execution mode: single or chain")


class AgentResult(BaseModel):
    """Model for agent execution results"""
    success: bool = Field(..., description="Whether the agent execution was successful")
    agent_used: str = Field(..., description="Name of the agent that executed")
    messages: List[str] = Field(default_factory=list, description="Response messages")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    execution_mode: str = Field(default="single", description="Execution mode used")


class OrchestrationResult(BaseModel):
    """Model for final orchestration results"""
    success: bool = Field(..., description="Overall success status")
    task: str = Field(..., description="Original task")
    selected_agents: List[str] = Field(default_factory=list, description="Agents that were selected")
    execution_plans: Dict[str, List[str]] = Field(default_factory=dict, description="Execution plans per agent")
    agent_results: Dict[str, AgentResult] = Field(default_factory=dict, description="Results from each agent")
    final_result: str = Field(default="", description="Final aggregated result")
    total_llm_calls: int = Field(default=0, description="Total LLM calls across all stages")


class ToolDefinition(BaseModel):
    """Model for tool definitions"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters schema")


class AgentCapabilities(BaseModel):
    """Model for agent capabilities"""
    name: str = Field(..., description="Agent name")
    capabilities: List[str] = Field(default_factory=list, description="List of agent capabilities")
    tools: List[ToolDefinition] = Field(default_factory=list, description="Available tools")
    status: str = Field(default="healthy", description="Agent health status")


class AgentProcessResult(BaseModel):
    success: bool
    messages: Optional[List[str]] = None
    agent_used: Optional[str] = None
    llm_calls: int = 0
    execution_mode: Optional[str] = None
    error: Optional[str] = None


