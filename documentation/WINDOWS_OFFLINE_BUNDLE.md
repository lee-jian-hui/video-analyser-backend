# Windows Offline Bundle: Tauri + React UI, Python gRPC Backend, and Ollama

This guide explains how to package the React + Tauri frontend, the Python gRPC backend, and a local Ollama LLM into a single Windows desktop app that runs fully offline.

Targets: Windows 10/11 x86_64, no internet required at runtime.

## Overview

- Frontend: React app bundled by Tauri
- Backend: Python gRPC server packaged as a single Windows executable (sidecar)
- LLM service: Ollama executable + preloaded models, started locally
- Offline ML assets: Whisper (speech-to-text), YOLO (vision) model weights

The Tauri app launches both the Python backend and Ollama as sidecars, then connects to the backend over `localhost`.

## Prerequisites (Build Machine)

- Node.js 18+ and pnpm/npm/yarn
- Rust toolchain (stable), Tauri CLI (`cargo install tauri-cli`)
- Visual Studio Build Tools (MSVC) for Windows
- Python 3.10+ with pip
- PyInstaller (`pip install pyinstaller`) or Nuitka (optional alternative)
- NSIS (for Tauri installer bundling on Windows)

Optional for GPU acceleration:
- NVIDIA CUDA/cuDNN installed and compatible PyTorch if your models use GPU

## Project Layout Assumptions

This repo contains your Python backend. Your Tauri project (Rust + frontend) lives alongside or in a sibling directory. We will refer to them as:

- `video-analyser-backend/` (this repo)
- `video-analyser-app/` (Tauri + React app)

You can adjust paths to match your setup.

## Step 1: Prepare Offline ML Assets

The backend expects local model files for Whisper and YOLO. Use the included script to fetch them on a machine with internet, then freeze into the app.

1) Download models into `ml-models/` in the backend repo:

```
cd video-analyser-backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py
```

This should populate something like:

```
ml-models/
  whisper/  # whisper *.pt
  yolo/     # yolov8*.pt (if your script stores here)
```

2) Verify your backend is configured to read from `ML_MODEL_CACHE_DIR`:

- `configs.Config.get_ml_model_cache_dir()` defaults to `./ml-models` when not bundled. We will ship this folder and point to it at runtime.

## Step 2: Prepare Ollama Offline

You need the Ollama Windows binary and the models you intend to use (e.g., `qwen3:0.6b`).

1) Install Ollama on a connected machine (Windows installer), then pull required models:

```
ollama pull qwen3:0.6b
```

2) Export the model to a portable file:

```
ollama show qwen3:0.6b --modelfile > qwen3_0.6b.Modelfile
ollama save qwen3:0.6b qwen3_0.6b.ollama
```

3) Collect files to ship in your Tauri resources:

- `ollama.exe` (Windows binary)
- A models directory (we’ll call it `ollama_models/`) containing `qwen3_0.6b.ollama`

At runtime, we will run:

```
ollama serve --host 127.0.0.1
ollama import qwen3_0.6b.ollama
```

Note: You can also pre-expand models by starting Ollama once with `OLLAMA_MODELS` pointing to your packaged models directory.

## Step 3: Package the Python Backend (gRPC)

We will create a single-file Windows executable using PyInstaller that includes your Python code, protobuf stubs, and templates.

1) Create a PyInstaller spec or use a command-line build. Example command:

```
cd video-analyser-backend
.venv\Scripts\activate
pyinstaller \
  --name video_analyzer_backend \
  --onefile \
  --noconsole \
  --add-data "templates;templates" \
  --add-data "protos;protos" \
  --add-data "ml-models;ml-models" \
  --hidden-import langchain \
  --hidden-import tiktoken \
  --hidden-import whisper \
  --hidden-import ultralytics \
  server.py
```

2) Verify the exe runs offline:

```
dist\video_analyzer_backend.exe
```

To ensure offline behavior, configure environment variables (see Step 5) so the backend uses only local/Ollama resources.

Tips:
- If you hit import errors at runtime, add missing packages via `--hidden-import`.
- If models are large, consider `--onefile` with care; alternatively use `--onedir` and ship a folder. Tauri sidecars support both.

## Step 4: Wire Sidecars in Tauri

In your `video-analyser-app` (Tauri project), configure sidecars for both the backend and Ollama. Example `tauri.conf.json` snippet:

```json
{
  "bundle": {
    "identifier": "com.example.videoanalyzer",
    "active": true,
    "targets": ["nsis"],
    "resources": [
      "resources/ml-models/**",
      "resources/ollama_models/**"
    ]
  },
  "tauri": {
    "allowlist": { "shell": { "all": true } },
    "windows": [{ "title": "Video Analyzer", "fullscreen": false }],
    "systemTray": { "iconPath": "icons/icon.ico" },
    "updater": { "active": false },
    "security": { "csp": null },
    "sidecar": [
      {
        "name": "video_analyzer_backend",
        "path": "sidecars/video_analyzer_backend.exe",
        "env": {
          "ORCHESTRATION_TOTAL_TIMEOUT_S": "180",
          "PER_AGENT_DEFAULT_BUDGET_S": "60",
          "FUNCTION_CALLING_BACKEND": "ollama",
          "CHAT_BACKEND": "ollama",
          "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
          "HF_HUB_OFFLINE": "true",
          "TRANSFORMERS_OFFLINE": "true",
          "ML_MODEL_CACHE_DIR": "{resources}/ml-models"
        }
      },
      {
        "name": "ollama",
        "path": "sidecars/ollama.exe",
        "env": {
          "OLLAMA_MODELS": "{resources}/ollama_models"
        },
        "args": ["serve", "--host", "127.0.0.1"]
      }
    ]
  }
}
```

Notes:
- `{resources}` resolves to the Tauri app resources directory at runtime. Place `ml-models` and `ollama_models` there.
- If you prefer, start Ollama first, perform one-time import of `.ollama` files, then launch the backend.

### Tauri Startup Logic (Rust)

In your Tauri main.rs (or a setup hook):

1) Spawn Ollama sidecar; wait until `127.0.0.1:11434` responds.
2) Spawn backend sidecar; wait for gRPC port (e.g., 50051) to become ready.
3) Launch the UI once both are healthy.

Consider implementing a retry with timeout and showing user-friendly status in the splash screen.

## Step 5: Offline Configuration

The backend must avoid network access at runtime. Recommended envs passed to the backend sidecar:

- `FUNCTION_CALLING_BACKEND=ollama`
- `CHAT_BACKEND=ollama`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `HF_HUB_OFFLINE=true`
- `TRANSFORMERS_OFFLINE=true`
- `ML_MODEL_CACHE_DIR={resources}/ml-models`

If you also use Whisper/YOLO:
- Ensure `ml-models/whisper/*.pt` and YOLO weights are present

## Step 6: Place Artifacts into Tauri App

Create the `video-analyser-app/src-tauri/sidecars/` and `resources/` folders and copy in:

- `dist/video_analyzer_backend.exe` → `src-tauri/sidecars/video_analyzer_backend.exe`
- `ollama.exe` → `src-tauri/sidecars/ollama.exe`
- `ml-models/` → `src-tauri/resources/ml-models/`
- `ollama_models/` (with your `.ollama` files) → `src-tauri/resources/ollama_models/`

Ensure your `tauri.conf.json` resource globs include these paths.

## Step 7: Build the Windows Installer

From the Tauri app directory:

```
cd video-analyser-app
pnpm install  # or npm/yarn
pnpm build    # builds the React frontend
cargo tauri build
```

This produces an NSIS installer containing the app, the two sidecar executables, and resources.

## Step 8: Validate Offline

On a clean Windows VM with no internet:

- Install and run the app
- Confirm Ollama process is running and reachable on 11434
- Confirm backend gRPC server is started (logs, or ping a health RPC)
- Run a simple vision task and a short transcription to verify Whisper/YOLO assets
- Confirm chat uses Ollama with the shipped model (no downloads attempted)

## Troubleshooting

- Missing DLLs when launching sidecars: ensure MSVC redistributables (installed by VS Build Tools) are present.
- PyInstaller import errors: add `--hidden-import` flags or switch to a `.spec` file for fine-grained control.
- Large installer size: consider smaller models (e.g., `qwen3:0.6b`, `whisper base`) and pruning debug symbols.
- Ollama model not found: verify `OLLAMA_MODELS` path and that the model is imported at first run.
- GPU issues: default to CPU for reliability; add a settings toggle if needed.

## Security and Signing

- Code-sign the sidecar executables and Tauri app for better Windows SmartScreen compatibility.
- Bind Ollama to `127.0.0.1` and use a random high port if preferred; never expose externally.

## Optional: One-Time Model Import Flow

On first launch, if no Ollama models exist in `OLLAMA_MODELS`, the app can:

1) Copy `.ollama` tarballs from `{resources}/ollama_models` to a writable app data directory
2) Run `ollama import *.ollama`
3) Continue startup once `ollama list` shows the models

This supports keeping the installer smaller if you compress models as `.ollama` files.

---

With the above, your Windows app ships as a fully offline desktop experience: UI (Tauri), Python gRPC backend, and a local LLM via Ollama—all bundled and launched together.

