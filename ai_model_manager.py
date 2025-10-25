import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import hashlib
from configs import Config

class AIModelManager:
    """Manages AI model downloads and caching for all agents"""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = Config.get_ml_model_cache_dir()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Model configurations
        self.model_configs = {
            "whisper": {
                "models": ["tiny", "base", "small", "medium", "large"],
                "default": "base",
                "cache_dir": self.models_dir / "whisper"
            },
            "yolo": {
                "models": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                "default": "yolov8n.pt",
                "cache_dir": self.models_dir / "yolo"
            }
        }

        # Create subdirectories
        for config in self.model_configs.values():
            config["cache_dir"].mkdir(exist_ok=True)

    def initialize_all_models(self) -> Dict[str, bool]:
        """Initialize all required models. Returns status for each model type."""
        results = {}

        self.logger.info("Initializing AI models...")

        # Initialize Whisper
        results["whisper"] = self._initialize_whisper()

        # Initialize YOLO
        results["yolo"] = self._initialize_yolo()

        return results

    def _initialize_whisper(self) -> bool:
        """Initialize Whisper model"""
        try:
            import whisper

            model_name = self.model_configs["whisper"]["default"]
            cache_dir = self.model_configs["whisper"]["cache_dir"]

            self.logger.info(f"Loading Whisper model: {model_name}")

            # Set Whisper cache directory
            os.environ["WHISPER_CACHE_DIR"] = str(cache_dir)

            # Load model (will download if not cached)
            model = whisper.load_model(model_name, download_root=str(cache_dir))

            self.logger.info(f"Whisper model {model_name} loaded successfully")
            return True

        except ImportError:
            self.logger.error("Whisper not installed. Run: pip install openai-whisper")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper: {e}")
            return False

    def _initialize_yolo(self) -> bool:
        """Initialize YOLO model"""
        try:
            from ultralytics import YOLO

            model_name = self.model_configs["yolo"]["default"]
            cache_dir = self.model_configs["yolo"]["cache_dir"]
            model_path = cache_dir / model_name

            self.logger.info(f"Loading YOLO model: {model_name}")

            # Load model (will download if not present)
            if not model_path.exists():
                # Download to our cache directory
                model = YOLO(model_name)  # This downloads to default location
                # Move to our cache directory if needed
                self._ensure_yolo_in_cache(model_name, cache_dir)
            else:
                model = YOLO(str(model_path))

            self.logger.info(f"YOLO model {model_name} loaded successfully")
            return True

        except ImportError:
            self.logger.error("Ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO: {e}")
            return False

    def _ensure_yolo_in_cache(self, model_name: str, cache_dir: Path):
        """Ensure YOLO model is in our cache directory"""
        import shutil
        from ultralytics.utils import ASSETS

        # Default ultralytics cache location
        default_cache = Path.home() / ".cache" / "ultralytics"
        default_model_path = default_cache / model_name

        our_model_path = cache_dir / model_name

        if default_model_path.exists() and not our_model_path.exists():
            shutil.copy2(default_model_path, our_model_path)
            self.logger.info(f"Copied {model_name} to local cache: {our_model_path}")

    def get_whisper_model(self, model_size: str = None) -> Optional[Any]:
        """Get cached Whisper model"""
        try:
            import whisper

            model_size = model_size or self.model_configs["whisper"]["default"]
            cache_dir = self.model_configs["whisper"]["cache_dir"]

            # Set cache directory
            os.environ["WHISPER_CACHE_DIR"] = str(cache_dir)

            return whisper.load_model(model_size, download_root=str(cache_dir))

        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            return None

    def get_yolo_model(self, model_size: str = None) -> Optional[Any]:
        """Get cached YOLO model"""
        try:
            from ultralytics import YOLO

            model_size = model_size or self.model_configs["yolo"]["default"]
            cache_dir = self.model_configs["yolo"]["cache_dir"]
            model_path = cache_dir / model_size

            if model_path.exists():
                return YOLO(str(model_path))
            else:
                # Fallback to download
                self.logger.warning(f"Model {model_size} not in cache, downloading...")
                return YOLO(model_size)

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return None

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}

        # Whisper status
        whisper_cache = self.model_configs["whisper"]["cache_dir"]
        whisper_files = list(whisper_cache.glob("*.pt")) if whisper_cache.exists() else []
        status["whisper"] = {
            "cache_dir": str(whisper_cache),
            "models_cached": [f.stem for f in whisper_files],
            "default_model": self.model_configs["whisper"]["default"]
        }

        # YOLO status
        yolo_cache = self.model_configs["yolo"]["cache_dir"]
        yolo_files = list(yolo_cache.glob("*.pt")) if yolo_cache.exists() else []
        status["yolo"] = {
            "cache_dir": str(yolo_cache),
            "models_cached": [f.name for f in yolo_files],
            "default_model": self.model_configs["yolo"]["default"]
        }

        return status

    def cleanup_old_models(self, keep_default: bool = True):
        """Clean up old or unused models"""
        if keep_default:
            self.logger.info("Cleanup with default models preserved is not implemented yet")
        else:
            self.logger.info("Full cleanup is not implemented yet")


# Global instance
_model_manager = None

def get_model_manager() -> AIModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = AIModelManager()
    return _model_manager

def initialize_models() -> Dict[str, bool]:
    """Initialize all models - call this at application startup"""
    manager = get_model_manager()
    return manager.initialize_all_models()