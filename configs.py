import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration management"""

    # fetch or downloads cached ml models at this directory
    @staticmethod
    def get_ml_model_cache_dir() -> str:
        """Get appropriate ML model cache directory based on environment"""
        import sys
        from pathlib import Path

        # Check if we're in a bundled/production environment
        if getattr(sys, 'frozen', False) or os.getenv("TAURI_ENV"):
            # Bundled app - use bundled models directory (read-only)
            if getattr(sys, 'frozen', False):
                # PyInstaller bundle
                bundle_dir = Path(sys._MEIPASS) / "ml-models"
            else:
                # Tauri bundle - models are in resources directory
                # Tauri puts resources next to the executable
                exe_dir = Path(os.path.dirname(sys.executable))
                bundle_dir = exe_dir / "ml-models"

            print(f"Using bundled models at {str(bundle_dir)}")
            return str(bundle_dir)

        else:
            # pment - use local directory
            return os.getenv("ML_MODEL_CACHE_DIR", "./ml-models")

    EXECUTION_MODE = "single"
    ML_MODEL_CACHE_DIR: str = get_ml_model_cache_dir()

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", None)
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )

    # Model Configuration - Unified Backend System
    # MODEL_BACKEND options: "ollama", "local", "remote"
    # Separate backends for function calling and chat
    FUNCTION_CALLING_BACKEND: str = os.getenv("FUNCTION_CALLING_BACKEND", "remote").lower()
    CHAT_BACKEND: str = os.getenv("CHAT_BACKEND", "remote").lower()

    # Remote (Cloud API) Configuration
    REMOTE_PROVIDER: str = os.getenv("REMOTE_PROVIDER", "google_genai")
    REMOTE_MODEL_NAME: str = os.getenv("REMOTE_MODEL_NAME", "gemini-2.0-flash-lite")
    REMOTE_TEMPERATURE: float = float(os.getenv("REMOTE_TEMPERATURE", "0.0"))

    # Ollama (Served Local) Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_FUNCTION_CALLING_MODEL: str = os.getenv("OLLAMA_FUNCTION_CALLING_MODEL", "qwen3:0.6b")
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:0.6b")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

    # Local (Transformers Pipeline) Configuration
    LOCAL_FUNCTION_CALLING_MODEL: str = os.getenv("LOCAL_FUNCTION_CALLING_MODEL", "qwen3")  # "llama", "codellama", "qwen", "qwen3", "phi3"
    LOCAL_CHAT_MODEL: str = os.getenv("LOCAL_CHAT_MODEL", "qwen3")  # "llama", "codellama", "qwen", "qwen3", "phi3"
    LOCAL_TEMPERATURE: float = float(os.getenv("LOCAL_TEMPERATURE", "0.1"))

    # Legacy support (deprecated but still works)
    USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    USE_LOCAL_FUNCTION_CALLING: bool = os.getenv("USE_LOCAL_FUNCTION_CALLING", "false").lower() == "true"
    USE_LOCAL_CHAT: bool = os.getenv("USE_LOCAL_CHAT", "false").lower() == "true"
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "google_genai")  # Alias for REMOTE_PROVIDER
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")  # Alias for REMOTE_MODEL_NAME
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
    FUNCTION_CALLING_MODEL_TYPE: str = os.getenv("FUNCTION_CALLING_MODEL_TYPE", "gemini")
    CHAT_MODEL_TYPE: str = os.getenv("CHAT_MODEL_TYPE", "phi3")

    # Local Model Hardware Configuration
    DEVICE_MAP: str = os.getenv("DEVICE_MAP", "cpu")  # "cpu", "auto", "cuda", etc.
    TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "auto")  # "auto", "float16", "float32"
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    INFERENCE_TIMEOUT: int = int(os.getenv("INFERENCE_TIMEOUT", "720"))  # seconds

    # HuggingFace Offline Configuration
    HF_HUB_OFFLINE: bool = os.getenv("HF_HUB_OFFLINE", "false").lower() == "true"
    TRANSFORMERS_OFFLINE: bool = os.getenv("TRANSFORMERS_OFFLINE", "false").lower() == "true"

    # Agent Configuration
    DEFAULT_EXECUTION_MODE: str = os.getenv("DEFAULT_EXECUTION_MODE", "single")
    MAX_LLM_CALLS: int = int(os.getenv("MAX_LLM_CALLS", "10"))

    # Video Processing Configuration
    YOLO_MODEL_SIZE: str = os.getenv("YOLO_MODEL_SIZE", "yolov8n")
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
    VIDEO_SAMPLE_INTERVAL: int = int(os.getenv("VIDEO_SAMPLE_INTERVAL", "30"))

    # Intent classification
    INTENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.5"))
    USE_INTENT_ROUTING: bool = os.getenv("USE_INTENT_ROUTING", "false").lower() == "true"

    # Orchestrator Configuration
    ENABLE_WORKFLOW_VISUALIZATION: bool = os.getenv("ENABLE_WORKFLOW_VISUALIZATION", "true").lower() == "true"
    ORCHESTRATOR_TIMEOUT: int = int(os.getenv("ORCHESTRATOR_TIMEOUT", "300"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))  # Maximum retries for agent selection and planning

    # Orchestration Time Budgets
    # Overall call-level timeout (seconds)
    ORCHESTRATION_TOTAL_TIMEOUT_S: float = float(os.getenv("ORCHESTRATION_TOTAL_TIMEOUT_S", "180"))
    # Default per-agent budget if no specific override (seconds)
    PER_AGENT_DEFAULT_BUDGET_S: float = float(os.getenv("PER_AGENT_DEFAULT_BUDGET_S", "60"))
    # Safety margin reserved for cleanup/formatting (seconds)
    SCHEDULER_SAFETY_MARGIN_S: float = float(os.getenv("SCHEDULER_SAFETY_MARGIN_S", "2"))
    # Optional per-agent overrides via JSON string env var
    # Example: '{"vision_agent": 45, "transcription_agent": 120}'
    _AGENT_BUDGETS_ENV = os.getenv("AGENT_BUDGETS_S_JSON", "")
    try:
        import json as _json
        AGENT_BUDGETS_S = _json.loads(_AGENT_BUDGETS_ENV) if _AGENT_BUDGETS_ENV else {}
    except Exception:
        AGENT_BUDGETS_S = {}


    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        required_fields = []

        for field in required_fields:
            if not getattr(cls, field):
                raise ValueError(f"Required configuration {field} is missing")

        return True

    @classmethod
    def get_model_config(cls) -> dict:
        """Get model configuration dictionary"""
        return {
            "model_name": cls.MODEL_NAME,
            "model_provider": cls.MODEL_PROVIDER,
            "temperature": cls.MODEL_TEMPERATURE,
            "api_key": cls.GEMINI_API_KEY
        }

    @classmethod
    def get_logging_config(cls) -> dict:
        """Get logging configuration dictionary"""
        return {
            "level": cls.LOG_LEVEL,
            "log_file": cls.LOG_FILE,
            "format_string": cls.LOG_FORMAT
        }

    @classmethod
    def get_video_config(cls) -> dict:
        """Get video processing configuration"""
        return {
            "yolo_model_size": cls.YOLO_MODEL_SIZE,
            "ocr_language": cls.OCR_LANGUAGE,
            "sample_interval": cls.VIDEO_SAMPLE_INTERVAL
        }


    @classmethod
    def print_config(cls):
        """Print current configuration (excluding sensitive data)"""
        from utils.logger import get_logger
        logger = get_logger(__name__)

        logger.info("Current Configuration:")
        logger.info(f"  LOG_LEVEL: {cls.LOG_LEVEL}")
        logger.info(f"  MODEL_NAME: {cls.MODEL_NAME}")
        logger.info(f"  MODEL_TEMPERATURE: {cls.MODEL_TEMPERATURE}")
        logger.info(f"  DEFAULT_EXECUTION_MODE: {cls.DEFAULT_EXECUTION_MODE}")
        logger.info(f"  YOLO_MODEL_SIZE: {cls.YOLO_MODEL_SIZE}")
        logger.info(f"  GEMINI_API_KEY: {'*' * len(cls.GEMINI_API_KEY) if cls.GEMINI_API_KEY else 'Not set'}")


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.error(f"Configuration Error: {e}")
    logger.error("Please check your .env file and ensure all required variables are set.")
