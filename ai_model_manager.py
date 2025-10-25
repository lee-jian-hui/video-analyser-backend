import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import hashlib
from enum import Enum
from configs import Config


class AIModelType(Enum):
    """Enum for tracking AI model types by capability"""
    TRANSCRIPTION = "transcription"    # Whisper for speech-to-text
    OBJECT_DETECTION = "object_detection"  # YOLO for object detection
    LLM = "llm"                       # Language model (Llama/CodeLlama/Gemini)

class AIModelManager:
    """Manages AI model downloads and caching for all agents"""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = Config.get_ml_model_cache_dir()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Model configurations using capability-based enum
        self.model_configs = {
            AIModelType.TRANSCRIPTION: {
                "models": ["tiny", "base", "small", "medium", "large"],
                "default": "base",
                "cache_dir": self.models_dir / "whisper"
            },
            AIModelType.OBJECT_DETECTION: {
                "models": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                "default": "yolov8n.pt",
                "cache_dir": self.models_dir / "yolo"
            },
            AIModelType.LLM: {
                "llama": {
                    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
                    "cache_dir": self.models_dir / "llama3-2b"
                },
                "codellama": {
                    "model_name": "codellama/CodeLlama-7b-Instruct-hf",
                    "cache_dir": self.models_dir / "codellama-7b"
                }
            }
        }

        # Create subdirectories
        for model_type, config in self.model_configs.items():
            if model_type == AIModelType.LLM:
                # LLM has nested structure
                for llm_type, llm_config in config.items():
                    llm_config["cache_dir"].mkdir(exist_ok=True)
            else:
                # Other models have direct cache_dir
                config["cache_dir"].mkdir(exist_ok=True)

    def initialize_all_models(self) -> Dict[AIModelType, bool]:
        """Initialize all required models. Returns status for each model type."""
        results = {}

        self.logger.info("Initializing AI models...")

        # Initialize Transcription (Whisper)
        results[AIModelType.TRANSCRIPTION] = self._initialize_whisper()

        # Initialize Object Detection (YOLO)
        results[AIModelType.OBJECT_DETECTION] = self._initialize_yolo()

        # Initialize LLM based on config
        results[AIModelType.LLM] = self._initialize_local_llm()

        return results

    def _initialize_whisper(self) -> bool:
        """Initialize Whisper model"""
        try:
            import whisper

            model_name = self.model_configs[AIModelType.TRANSCRIPTION]["default"]
            cache_dir = self.model_configs[AIModelType.TRANSCRIPTION]["cache_dir"]

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
            from ultralytics.utils import SETTINGS

            model_name = self.model_configs[AIModelType.OBJECT_DETECTION]["default"]
            cache_dir = self.model_configs[AIModelType.OBJECT_DETECTION]["cache_dir"]
            model_path = cache_dir / model_name

            # Set ultralytics weights directory to our cache
            SETTINGS['weights_dir'] = str(cache_dir)

            self.logger.info(f"Loading YOLO model: {model_name}")

            # Load model (will download if not present)
            if not model_path.exists():
                # Download model - YOLO will download to its default cache
                model = YOLO(model_name)
                # Copy to our cache directory
                self._ensure_yolo_in_cache(model_name, cache_dir)
                # Verify the copy worked
                if model_path.exists():
                    self.logger.info(f"YOLO model {model_name} cached successfully")
                else:
                    self.logger.warning(f"Failed to cache YOLO model to {model_path}")
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

    def _initialize_llama(self) -> bool:
        """Initialize Llama model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_config = self.model_configs[AIModelType.LLM]["llama"]
            model_name = model_config["model_name"]
            cache_dir = model_config["cache_dir"]

            self.logger.info(f"Loading Llama model: {model_name}")

            # Download/load tokenizer and model to cache directory
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                local_files_only=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                dtype=Config.TORCH_DTYPE,
                device_map=Config.DEVICE_MAP,
                local_files_only=False
            )

            self.logger.info(f"Llama model {model_name} loaded successfully")
            return True

        except ImportError:
            self.logger.error("Transformers not installed. Run: pip install transformers torch")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize Llama: {e}")
            return False

    def _initialize_codellama(self) -> bool:
        """Initialize CodeLlama model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_config = self.model_configs[AIModelType.LLM]["codellama"]
            model_name = model_config["model_name"]
            cache_dir = model_config["cache_dir"]

            self.logger.info(f"Loading CodeLlama model: {model_name}")

            # Download/load tokenizer and model to cache directory
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                local_files_only=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                dtype=Config.TORCH_DTYPE,
                device_map=Config.DEVICE_MAP,
                local_files_only=False
            )

            self.logger.info(f"CodeLlama model {model_name} loaded successfully")
            return True

        except ImportError:
            self.logger.error("Transformers not installed. Run: pip install transformers torch")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize CodeLlama: {e}")
            return False

    def _initialize_local_llm(self) -> bool:
        """Initialize the configured local LLM"""
        local_model_type = Config.LOCAL_MODEL_TYPE.lower()

        if local_model_type == "codellama":
            return self._initialize_codellama()
        elif local_model_type == "llama":
            return self._initialize_llama()
        else:
            self.logger.error(f"Unknown local model type: {local_model_type}")
            return False

    def _ensure_yolo_in_cache(self, model_name: str, cache_dir: Path):
        """Ensure YOLO model is in our cache directory"""
        import shutil
        import glob

        our_model_path = cache_dir / model_name

        # Try multiple common ultralytics cache locations
        possible_locations = [
            Path.home() / ".cache" / "ultralytics" / model_name,
            Path.home() / ".ultralytics" / model_name,
            Path.cwd() / model_name,
        ]

        # Also try to find the model in any ultralytics directory
        ultralytics_cache = Path.home() / ".cache" / "ultralytics"
        if ultralytics_cache.exists():
            found_models = list(ultralytics_cache.glob(f"**/{model_name}"))
            possible_locations.extend(found_models)

        # Find the first existing model file
        for source_path in possible_locations:
            if source_path.exists() and source_path.is_file():
                try:
                    shutil.copy2(source_path, our_model_path)
                    self.logger.info(f"Copied {model_name} from {source_path} to {our_model_path}")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to copy {model_name}: {e}")

        self.logger.warning(f"Could not find YOLO model {model_name} to copy to cache")

    def get_whisper_model(self, model_size: str = None) -> Optional[Any]:
        """Get cached Whisper model"""
        try:
            import whisper

            model_size = model_size or self.model_configs[AIModelType.TRANSCRIPTION]["default"]
            cache_dir = self.model_configs[AIModelType.TRANSCRIPTION]["cache_dir"]

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
            from ultralytics.utils import SETTINGS

            model_size = model_size or self.model_configs[AIModelType.OBJECT_DETECTION]["default"]
            cache_dir = self.model_configs[AIModelType.OBJECT_DETECTION]["cache_dir"]
            model_path = cache_dir / model_size

            # Set ultralytics weights directory to our cache
            SETTINGS['weights_dir'] = str(cache_dir)

            if model_path.exists():
                return YOLO(str(model_path))
            else:
                # Fallback to download
                self.logger.warning(f"Model {model_size} not in cache, downloading...")
                return YOLO(model_size)

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return None

    def get_llama_model(self):
        """Get cached Llama model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_config = self.model_configs[AIModelType.LLM]["llama"]
            model_name = model_config["model_name"]
            cache_dir = model_config["cache_dir"]

            # Load from cache (will use cached files automatically)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir)
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                dtype="auto",
                device_map="cpu"
            )

            return {"model": model, "tokenizer": tokenizer}

        except Exception as e:
            self.logger.error(f"Failed to load Llama model: {e}")
            return None

    def get_codellama_model(self):
        """Get cached CodeLlama model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_config = self.model_configs[AIModelType.LLM]["codellama"]
            model_name = model_config["model_name"]
            cache_dir = model_config["cache_dir"]

            # Load from cache (will use cached files automatically)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir)
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                dtype="auto",
                device_map="cpu"
            )

            return {"model": model, "tokenizer": tokenizer}

        except Exception as e:
            self.logger.error(f"Failed to load CodeLlama model: {e}")
            return None

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}

        # Transcription status
        transcription_cache = self.model_configs[AIModelType.TRANSCRIPTION]["cache_dir"]
        transcription_files = list(transcription_cache.glob("*.pt")) if transcription_cache.exists() else []
        status["transcription"] = {
            "cache_dir": str(transcription_cache),
            "models_cached": [f.stem for f in transcription_files],
            "default_model": self.model_configs[AIModelType.TRANSCRIPTION]["default"]
        }

        # Object Detection status
        object_detection_cache = self.model_configs[AIModelType.OBJECT_DETECTION]["cache_dir"]
        object_detection_files = list(object_detection_cache.glob("*.pt")) if object_detection_cache.exists() else []
        status["object_detection"] = {
            "cache_dir": str(object_detection_cache),
            "models_cached": [f.name for f in object_detection_files],
            "default_model": self.model_configs[AIModelType.OBJECT_DETECTION]["default"]
        }

        # LLM status
        llm_configs = self.model_configs[AIModelType.LLM]
        status["llm"] = {}
        for llm_type, llm_config in llm_configs.items():
            llm_cache = llm_config["cache_dir"]
            status["llm"][llm_type] = {
                "cache_dir": str(llm_cache),
                "model_name": llm_config["model_name"],
                "cached": llm_cache.exists() and any(llm_cache.iterdir())
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