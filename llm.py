
from langchain.chat_models import init_chat_model


def get_model(api_key: str):
    """Initialize Gemini model with API key"""
    return init_chat_model(
        "gemini-2.0-flash-lite",
        model_provider="google_genai",
        temperature=0,
        api_key=api_key
    )



