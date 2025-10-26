# Model Configuration Guide

This project supports multiple model backends with separate configurations for function calling and chat models.

## Quick Start

### Use Ollama (Recommended for local deployment)
```bash
FUNCTION_CALLING_BACKEND=ollama
OLLAMA_FUNCTION_CALLING_MODEL=qwen3:0.6b
```

### Use Local Transformers Pipeline
```bash
FUNCTION_CALLING_BACKEND=local
LOCAL_FUNCTION_CALLING_MODEL=qwen3
```

### Use Remote API (Gemini)
```bash
FUNCTION_CALLING_BACKEND=remote
REMOTE_MODEL_NAME=gemini-2.0-flash-lite
GEMINI_API_KEY=your_api_key
```

## Configuration Options

### Backend Selection

**For Function Calling:**
- `FUNCTION_CALLING_BACKEND`: `ollama`, `local`, or `remote` (default: `remote`)

**For Chat:**
- `CHAT_BACKEND`: `ollama`, `local`, or `remote` (default: `remote`)

### Ollama Configuration

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_FUNCTION_CALLING_MODEL=qwen3:0.6b
OLLAMA_CHAT_MODEL=qwen3:0.6b
OLLAMA_TEMPERATURE=0.1
```

### Local (Transformers) Configuration

```bash
LOCAL_FUNCTION_CALLING_MODEL=qwen3  # Options: llama, codellama, qwen, qwen3, phi3
LOCAL_CHAT_MODEL=qwen3
LOCAL_TEMPERATURE=0.1
```

### Remote (Cloud API) Configuration

```bash
REMOTE_PROVIDER=google_genai
REMOTE_MODEL_NAME=gemini-2.0-flash-lite
REMOTE_TEMPERATURE=0.0
GEMINI_API_KEY=your_api_key_here
```

## Example Configurations

### All Ollama
```bash
FUNCTION_CALLING_BACKEND=ollama
CHAT_BACKEND=ollama
OLLAMA_FUNCTION_CALLING_MODEL=qwen3:0.6b
OLLAMA_CHAT_MODEL=qwen3:0.6b
```

### Hybrid: Ollama for function calling, Gemini for chat
```bash
FUNCTION_CALLING_BACKEND=ollama
CHAT_BACKEND=remote
OLLAMA_FUNCTION_CALLING_MODEL=qwen3:0.6b
REMOTE_MODEL_NAME=gemini-2.0-flash-lite
GEMINI_API_KEY=your_api_key
```

### All Local Transformers
```bash
FUNCTION_CALLING_BACKEND=local
CHAT_BACKEND=local
LOCAL_FUNCTION_CALLING_MODEL=qwen3
LOCAL_CHAT_MODEL=phi3
```

## Legacy Configuration (Still Supported)

The old configuration style still works for backward compatibility:

```bash
USE_OLLAMA=true  # Maps to FUNCTION_CALLING_BACKEND=ollama
USE_LOCAL_FUNCTION_CALLING=true  # Maps to FUNCTION_CALLING_BACKEND=local
USE_LOCAL_CHAT=true  # Maps to CHAT_BACKEND=local
```

## Backend Priority

When multiple config methods are used, the priority is:
1. Legacy flags (`USE_OLLAMA`, `USE_LOCAL_*`)
2. New backend config (`FUNCTION_CALLING_BACKEND`, `CHAT_BACKEND`)
3. Defaults (remote/Gemini)
