
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from configs import Config
from ai_model_manager import get_model_manager
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline


def get_llm_model() -> BaseChatModel:
    """Initialize model using configuration - backwards compatibility"""
    # For backwards compatibility, use function calling model
    return get_function_calling_llm()

def get_function_calling_llm() -> BaseChatModel:
    """Get LLM optimized for function calling and tool use"""
    # Handle legacy config first
    if Config.USE_OLLAMA:
        backend = "ollama"
    elif Config.USE_LOCAL_FUNCTION_CALLING or Config.USE_LOCAL_LLM:
        backend = "local"
    else:
        backend = Config.FUNCTION_CALLING_BACKEND

    # Route based on backend
    if backend == "ollama":
        return _get_ollama_model(model_type="function_calling")
    elif backend == "local":
        return _get_local_model(model_type="function_calling")
    elif backend == "remote":
        return _get_remote_model()
    else:
        raise ValueError(f"Unknown FUNCTION_CALLING_BACKEND: {backend}. Use 'ollama', 'local', or 'remote'")

def get_chat_llm() -> BaseChatModel:
    """Get LLM optimized for conversational responses"""
    # Handle legacy config first
    if Config.USE_LOCAL_CHAT:
        backend = "local"
    else:
        backend = Config.CHAT_BACKEND

    # Route based on backend
    if backend == "ollama":
        return _get_ollama_model(model_type="chat")
    elif backend == "local":
        return _get_local_model(model_type="chat")
    elif backend == "remote":
        return _get_remote_model()
    else:
        raise ValueError(f"Unknown CHAT_BACKEND: {backend}. Use 'ollama', 'local', or 'remote'")

def _get_remote_model() -> BaseChatModel:
    """Get remote cloud API model (Gemini, OpenAI, etc.)"""
    api_key = Config.GEMINI_API_KEY

    # Use new config or fallback to legacy
    provider = Config.REMOTE_PROVIDER or Config.MODEL_PROVIDER
    model_name = Config.REMOTE_MODEL_NAME or Config.MODEL_NAME
    temperature = Config.REMOTE_TEMPERATURE or Config.MODEL_TEMPERATURE

    return init_chat_model(
        model_name,
        model_provider=provider,
        temperature=temperature,
        api_key=api_key
    )

# Legacy alias
def _get_gemini_model():
    """Legacy function - use _get_remote_model() instead"""
    return _get_remote_model()


def _get_ollama_model(model_type: str = "function_calling") -> BaseChatModel:
    """Get Ollama-served model using ChatOllama

    Args:
        model_type: "function_calling" or "chat" to select appropriate model
    """
    try:
        from langchain_ollama import ChatOllama

        # Select model based on type
        if model_type == "function_calling":
            model_name = Config.OLLAMA_FUNCTION_CALLING_MODEL
        else:  # chat
            model_name = Config.OLLAMA_CHAT_MODEL

        print(f"Creating Ollama {model_type} model: {model_name}")
        print(f"Connecting to Ollama at: {Config.OLLAMA_BASE_URL}")

        chat_model = ChatOllama(
            model=model_name,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=Config.OLLAMA_TEMPERATURE,
        )

        return chat_model

    except Exception as e:
        print(f"Failed to initialize Ollama model: {e}")
        print("Falling back to remote model...")
        return _get_remote_model()


def _get_local_model(model_type: str = "function_calling") -> BaseChatModel:
    """Get local model using HuggingFace transformers pipeline

    Args:
        model_type: "function_calling" or "chat" to select appropriate model
    """
    # Select model based on type
    if model_type == "function_calling":
        model_name_config = (Config.LOCAL_FUNCTION_CALLING_MODEL or
                           Config.FUNCTION_CALLING_MODEL_TYPE or
                           Config.LOCAL_MODEL_TYPE).lower()
    else:  # chat
        model_name_config = (Config.LOCAL_CHAT_MODEL or
                           Config.CHAT_MODEL_TYPE or
                           Config.LOCAL_MODEL_TYPE).lower()

    try:
        # Get model manager and load the configured local model
        model_manager = get_model_manager()

        if model_name_config == "codellama":
            components = model_manager.get_codellama_model()
            model_name = "CodeLlama"
        elif model_name_config == "qwen":
            components = model_manager.get_qwen_1_5_b_model()
            model_name = "Qwen2.5-Coder-1.5B"
        elif model_name_config == "qwen3":
            components = model_manager.get_qwen3_1_7b_model()
            model_name = "Qwen3-1.7B"
        elif model_name_config == "phi3":
            components = model_manager.get_phi3_model()
            model_name = "Phi-3"
        else:
            components = model_manager.get_llama_model()
            model_name = "Llama"

        if components is None:
            raise Exception(f"Failed to load {model_name} model from cache")

        model = components["model"]
        tokenizer = components["tokenizer"]

        # Create text generation pipeline with config
        temperature = Config.LOCAL_TEMPERATURE
        print(f"Creating {model_name} pipeline with max_tokens={Config.MAX_NEW_TOKENS}, temperature={temperature}")
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=True,
            return_full_text=False
        )

        # Wrap in LangChain pipeline first
        pipeline_llm = HuggingFacePipeline(pipeline=text_pipeline)

        # Then wrap in ChatHuggingFace for chat model interface
        chat_model = ChatHuggingFace(llm=pipeline_llm)

        return chat_model

    except Exception as e:
        print(f"Failed to initialize local model: {e}")
        print("Falling back to remote model...")
        return _get_remote_model()


# Legacy function aliases
def _get_local_llm() -> BaseChatModel:
    """Legacy function - use _get_local_model() instead"""
    return _get_local_model()

def _get_local_function_calling_llm() -> BaseChatModel:
    """Legacy function - use _get_local_model() instead"""
    return _get_local_model()

def _get_local_chat_llm() -> BaseChatModel:
    """Get local LLM optimized for conversational responses"""
    try:
        model_manager = get_model_manager()
        
        # Use models good at conversation for chat
        chat_model_type = Config.CHAT_MODEL_TYPE.lower()
        
        if chat_model_type == "phi3":
            components = model_manager.get_phi3_model()
            model_name = "Phi-3"
        elif chat_model_type == "llama":
            components = model_manager.get_llama_model()
            model_name = "Llama"
        elif chat_model_type == "qwen":
            components = model_manager.get_qwen_1_5_b_model()
            model_name = "Qwen2.5-Coder-1.5B"
        elif chat_model_type == "qwen3":
            components = model_manager.get_qwen3_1_7b_model()
            model_name = "Qwen3-1.7B"
        elif chat_model_type == "gemini":
            return _get_gemini_model()
        else:
            components = model_manager.get_llama_model()
            model_name = "Llama"

        if components is None:
            raise Exception(f"Failed to load {model_name} model for chat")

        model = components["model"]
        tokenizer = components["tokenizer"]

        print(f"Creating {model_name} chat pipeline")
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=0.7,  # Higher temperature for creative responses
            do_sample=True,
            return_full_text=False
        )

        pipeline_llm = HuggingFacePipeline(pipeline=text_pipeline)
        chat_model = ChatHuggingFace(llm=pipeline_llm)
        return chat_model

    except Exception as e:
        print(f"Failed to initialize chat LLM: {e}")
        print("Falling back to Gemini model...")
        return _get_gemini_model()
