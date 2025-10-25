from langchain.messages import SystemMessage
from langchain.messages import ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from dotenv import load_dotenv
import os 


from graph import MessagesState
from tools import inject_llm_tools
from llm import get_model
from orchestrator import MultiStageOrchestrator
from utils.logger import get_logger
from models.task_models import VideoTask, TaskRequest

load_dotenv()

def initialize_model():
    """Initialize model with tools using the provided API key"""
    model = get_model(os.getenv("GEMINI_API_KEY"))
    return inject_llm_tools(model)


def create_llm_call(model_with_tools):
    """Create llm_call function with the model"""
    def llm_call(state: dict):
        """LLM decides whether to call a tool or not"""
        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }
    return llm_call



def create_tool_node(tools_by_name):
    """Create tool_node function with the tools"""
    def tool_node(state: dict):
        """Performs the tool call"""
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    return tool_node




def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END



def old_run():
    # Initialize model with tools
    model_with_tools, tools_by_name = initialize_model()

    # Create llm_call function with the model
    llm_call = create_llm_call(model_with_tools)

    # Create tool_node function with the tools
    tool_node = create_tool_node(tools_by_name)

    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()

    # Show the agent
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    # Invoke
    from langchain.messages import HumanMessage
    messages = [HumanMessage(content="Add 3 and 4.")]
    result = agent.invoke({"messages": messages})
    for m in result["messages"]:
        m.pretty_print()





def run():
    """New multi-stage orchestration entry point"""
    # Set up debug logging
    from utils.logger import setup_logging
    setup_logging(level="DEBUG")

    logger = get_logger(__name__)

    # Initialize the orchestrator
    orchestrator = MultiStageOrchestrator()

    logger.info("🚀 Multi-Stage LLM Orchestrator initialized!")
    logger.info(f"Available agents: {list(orchestrator.coordinator.get_available_agents().keys())}")

    # Create a VideoTask using Pydantic model
    video_task = VideoTask(
        description="Detect what objects are in the video",
        file_path="./sample.mp4",
        task_type="object_detection",
        output_format="summary",
        confidence_threshold=0.5
    )

    # Wrap in TaskRequest
    task_request = TaskRequest(
        task=video_task,
        execution_mode="chain"
    )

    # Alternative tasks (commented out):
    # VideoTask(description="Detect all people and objects and tell me what they are", file_path="./sample.mp4", task_type="object_detection")
    # VideoTask(description="Comprehensive analysis with objects and text", file_path="./sample.mp4", task_type="comprehensive_analysis")

    logger.info(f"\n📋 Processing task: {video_task.get_task_description()}")
    logger.info(f"📁 File path: {video_task.file_path}")
    logger.info(f"🎯 Task type: {video_task.task_type}")
    logger.info(f"⚙️ Execution mode: {task_request.execution_mode}")
    logger.info("\n" + "="*80)

    # Process through multi-stage orchestration
    result = orchestrator.process_task(task_request)

    logger.info("\n📊 ORCHESTRATION RESULTS:")
    logger.info("="*80)
    logger.info(f"✅ Success: {result['success']}")
    logger.info(f"🤖 Selected Agents: {result['selected_agents']}")
    logger.info(f"🛠️  Execution Plans: {result['execution_plans']}")
    logger.info(f"🧠 Total LLM Calls: {result['total_llm_calls']}")
    logger.info(f"\n📄 Final Result:\n{result['final_result']}")

    # Show workflow visualization
    try:
        from IPython.display import Image, display
        logger.info("\n📈 Workflow Visualization:")
        display(Image(orchestrator.workflow.get_graph().draw_mermaid_png()))
    except:
        logger.warning("Workflow visualization not available")

if __name__ == "__main__":
    run()