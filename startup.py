"""
Application startup module - initializes all AI models on first run
"""

import logging
from ai_model_manager import initialize_models, get_model_manager
from utils.logger import get_logger


def startup_initialization():
    """
    Run this at application startup to ensure all models are downloaded and ready.
    Returns True if successful, False if critical models failed to load.
    """
    logger = get_logger(__name__)
    logger.info("Starting application initialization...")

    # Initialize all models
    model_status = initialize_models()

    # Log results
    for model_type, success in model_status.items():
        if success:
            logger.info(f"✓ {model_type} model initialized successfully")
        else:
            logger.error(f"✗ {model_type} model failed to initialize")

    # Check if critical models loaded
    critical_models = ["whisper", "yolo"]
    failed_critical = [model for model in critical_models if not model_status.get(model, False)]

    if failed_critical:
        logger.error(f"Critical models failed: {failed_critical}")
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