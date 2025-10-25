"""
Application startup module - initializes all AI models on first run
"""

import logging
from agent_manager import get_agent_manager
from ai_model_manager import initialize_models, get_model_manager
from utils.logger import get_logger


def startup_initialization():
    """
    Run this at application startup for health checks and agent setup.
    Downloads models if needed and checks which services are available.
    """
    logger = get_logger(__name__)
    logger.info("Starting application initialization...")

    # Get agent manager
    agent_manager = get_agent_manager()

    # Download models if needed (for bundling or first run)
    logger.info("Ensuring models are downloaded...")
    model_status = initialize_models()

    # Run health checks
    logger.info("Running health checks...")
    health_status = agent_manager.startup_health_check()

    # Log results
    logger.info("=== Model Health Status ===")
    for model_type, healthy in health_status.items():
        status = "✓ HEALTHY" if healthy else "✗ UNAVAILABLE"
        logger.info(f"  {model_type}: {status}")

    # Log agent availability
    logger.info("=== Agent Availability ===")
    agent_status = agent_manager.get_agent_status()
    for agent_name, status in agent_status.items():
        agent_state = status['status'].upper()
        deps = ', '.join(status['dependencies'])
        logger.info(f"  {agent_name}: {agent_state} (deps: {deps})")

    available_agents = agent_manager.get_available_agents()
    logger.info(f"Available agents: {available_agents}")

    if not available_agents:
        logger.error("No agents available! Check model dependencies.")
        return False

    logger.info("Application initialization complete!")
    return True


def get_model_status_report():
    """Get detailed status report of all models"""
    manager = get_model_manager()
    return manager.get_model_status()


if __name__ == "__main__":
    # Run standalone for testing
    logging.basicConfig(level=logging.INFO)
    success = startup_initialization()

    if success:
        print("✓ All models initialized successfully")
        print("\nModel Status:")
        import json
        print(json.dumps(get_model_status_report(), indent=2))
    else:
        print("✗ Some models failed to initialize")
        exit(1)