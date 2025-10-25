#!/usr/bin/env python3
"""
Build-time script to download all required AI models
Run this during development or CI/CD to prepare models for bundling
"""

import os
import sys
from pathlib import Path
from ai_model_manager import AIModelManager
from utils.logger import get_logger

def download_all_models():
    """Download all required models for bundling"""
    logger = get_logger(__name__)

    # Force use local directory for downloading
    models_dir = "./ml-models"
    logger.info(f"Downloading models to: {models_dir}")

    # Create model manager with explicit local directory
    manager = AIModelManager(models_dir=models_dir)

    # Download all models
    results = manager.initialize_all_models()

    # Report results
    success_count = sum(results.values())
    total_count = len(results)

    logger.info(f"Downloaded {success_count}/{total_count} model types successfully")

    if success_count == total_count:
        logger.info("✓ All models ready for bundling!")
        return True
    else:
        failed = [model for model, success in results.items() if not success]
        logger.error(f"✗ Failed to download: {failed}")
        return False

def verify_models():
    """Verify all models are present and loadable"""
    logger = get_logger(__name__)
    models_dir = Path("./ml-models")

    if not models_dir.exists():
        logger.error("Models directory not found!")
        return False

    # Check Whisper models
    whisper_dir = models_dir / "whisper"
    whisper_models = list(whisper_dir.glob("*.pt")) if whisper_dir.exists() else []

    # Check YOLO models
    yolo_dir = models_dir / "yolo"
    yolo_models = list(yolo_dir.glob("*.pt")) if yolo_dir.exists() else []

    logger.info(f"Found Whisper models: {[m.name for m in whisper_models]}")
    logger.info(f"Found YOLO models: {[m.name for m in yolo_models]}")

    # Minimum requirements
    has_whisper = len(whisper_models) > 0
    has_yolo = len(yolo_models) > 0

    if has_whisper and has_yolo:
        logger.info("✓ All required models present")
        return True
    else:
        logger.error("✗ Missing required models")
        return False

def main():
    """Main entry point"""
    import logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        success = verify_models()
    else:
        success = download_all_models()

    if not success:
        sys.exit(1)

    print("Models ready for bundling!")

if __name__ == "__main__":
    main()