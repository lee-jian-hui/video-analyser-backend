
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from configs import Config
from ai_model_manager import get_model_manager
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline


def get_llm_model() -> BaseChatModel:
    """Initialize model using configuration"""

    # Check if we should use local LLM
    if Config.USE_LOCAL_LLM:
        return _get_local_llm()
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
        else:
            components = model_manager.get_llama_model()
            model_name = "Llama"

        if components is None:
            raise Exception(f"Failed to load {model_name} model from cache")

        model = components["model"]
        tokenizer = components["tokenizer"]

        # Create text generation pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
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
