"""Test Ollama integration with the main codebase"""

import os
# New config style
os.environ["FUNCTION_CALLING_BACKEND"] = "ollama"
os.environ["OLLAMA_FUNCTION_CALLING_MODEL"] = "qwen3:0.6b"

from llm import get_function_calling_llm
from langchain.messages import HumanMessage

# Get the Ollama model
print("Initializing Ollama model...")
model = get_function_calling_llm()

# Test basic chat
print("\n=== Testing Basic Chat ===")
messages = [HumanMessage(content="What is 15 + 27?")]
response = model.invoke(messages)
print(f"Response content: {response.content}")
print(f"Response type: {type(response)}")
print(f"Full response: {response}")
if hasattr(response, 'response_metadata'):
    print(f"Metadata: {response.response_metadata}")

# Test with vision agent tools
print("\n=== Testing with Tools ===")
from agents.vision_agent import detect_objects_in_video, extract_text_from_video

# Bind tools to the model
tools = [detect_objects_in_video, extract_text_from_video]
model_with_tools = model.bind_tools(tools)

# Test tool calling
messages = [HumanMessage(content="I have a video file at sample.mp4. What objects are in it?")]
response = model_with_tools.invoke(messages)

print(f"Response content: {response.content}")
print(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}")

print("\n=== Test Complete ===")
