
from langchain.chat_models import init_chat_model
from configs import Config


def get_model(api_key: str = None):
    """Initialize model using configuration"""
    api_key = api_key or Config.GEMINI_API_KEY
    model_config = Config.get_model_config()

    return init_chat_model(
        model_config["model_name"],
        model_provider=model_config["model_provider"],
        temperature=model_config["temperature"],
        api_key=api_key
    )



