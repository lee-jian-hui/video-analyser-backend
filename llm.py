
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
    if Config.USE_LOCAL_FUNCTION_CALLING:
        return _get_local_function_calling_llm()
    else:
        return _get_gemini_model()

def get_chat_llm() -> BaseChatModel:
    """Get LLM optimized for conversational responses"""
    if Config.USE_LOCAL_CHAT:
        return _get_local_chat_llm()
    else:
        return _get_gemini_model()

def _get_gemini_model():
    api_key = Config.GEMINI_API_KEY
    model_config = Config.get_model_config()

    return init_chat_model(
        model_config["model_name"],
        model_provider=model_config["model_provider"],
        temperature=model_config["temperature"],
        api_key=api_key
    )


def _get_local_llm() -> BaseChatModel:
    """Get local Llama model using HuggingFace transformers"""

    try:
        # Get model manager and load the configured local model
        model_manager = get_model_manager()

        if Config.LOCAL_MODEL_TYPE.lower() == "codellama":
            components = model_manager.get_codellama_model()
            model_name = "CodeLlama"
        elif Config.LOCAL_MODEL_TYPE.lower() == "qwen":
            components = model_manager.get_qwen_1_5_b_model()
            model_name = "Qwen"
        elif Config.LOCAL_MODEL_TYPE.lower() == "phi3":
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
        print(f"Creating {model_name} pipeline with max_tokens={Config.MAX_NEW_TOKENS}")
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )

        # Wrap in LangChain pipeline first
        pipeline_llm = HuggingFacePipeline(pipeline=text_pipeline)

        # Then wrap in ChatHuggingFace for chat model interface
        chat_model = ChatHuggingFace(llm=pipeline_llm)

        return chat_model

    except Exception as e:
        print(f"Failed to initialize local LLM: {e}")
        print("Falling back to Gemini model...")
        return _get_gemini_model()

def _get_local_function_calling_llm() -> BaseChatModel:
    """Get local LLM optimized for function calling (structured reasoning)"""
    try:
        model_manager = get_model_manager()
        
        # Use models good at structured output for function calling
        function_model_type = Config.FUNCTION_CALLING_MODEL_TYPE.lower()
        
        if function_model_type == "codellama":
            components = model_manager.get_codellama_model()
            model_name = "CodeLlama"
        elif function_model_type == "qwen":
            components = model_manager.get_qwen_1_5_b_model()
            model_name = "Qwen"
        elif function_model_type == "gemini":
            return _get_gemini_model()
        else:
            components = model_manager.get_llama_model()
            model_name = "Llama"

        if components is None:
            raise Exception(f"Failed to load {model_name} model for function calling")

        model = components["model"]
        tokenizer = components["tokenizer"]

        print(f"Creating {model_name} function calling pipeline")
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=0.1,  # Lower temperature for structured output
            do_sample=True,
            return_full_text=False
        )

        pipeline_llm = HuggingFacePipeline(pipeline=text_pipeline)
        chat_model = ChatHuggingFace(llm=pipeline_llm)
        return chat_model

    except Exception as e:
        print(f"Failed to initialize function calling LLM: {e}")
        print("Falling back to Gemini model...")
        return _get_gemini_model()

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
            model_name = "Qwen"
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
