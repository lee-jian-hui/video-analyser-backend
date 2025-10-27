## ENVIRONMENT
- WSL Linux Ubuntu 22.04
- Python 3.10.12

## video files used in testing:
- https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4
- https://file-examples.com/wp-content/storage/2017/04/file_example_MP4_480_1_5MG.mp4

## PRE-REQ

### Python Dependencies
- have `pip` or `uv` to install dependencies
- `pip install uv` to get uv which is a python package manager
- `cp sample_env.txt`

### System Dependencies
- **ffmpeg** - Required for video/audio processing (transcription agent)

Install on Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Install on macOS:
```bash
brew install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```


## LOCAL SETUP
```
# use a dependency manager like conda 
# i use uv here
uv sync
python download.py
python main.py
```



## start python grpc server
```
cd ./python-backend
python server.py
```

## Testing the grpc endpoints
```
grpcurl -plaintext -proto protos/video_analyzer.proto localhost:50051 list
grpcurl -plaintext -proto protos/video_analyzer.proto describe video_analyzer.VideoAnalyzerService
grpcurl -plaintext -proto protos/video_analyzer.proto \
  -d '{"message": "hello from grpcurl"}' \
  localhost:50051 \
  video_analyzer.VideoAnalyzerService/SendChatMessage
```

### Uploading a video for testing
Streaming uploads are easier with the helper script under `scripts/upload_video.py`:
```
uv run python scripts/upload_video.py /path/to/video.mp4 \
  --address localhost:50051 \
  --chunk-size 1048576
```
The script streams the file in chunks, prints the returned `file_id`, and you can reuse that ID with `SendChatMessage`.

### Registering an existing local file (desktop flow)
When the frontend can share the absolute file path, skip uploading altogether and let the backend register it:
```python
from services.video_registrar import VideoRegistrar
from services.video_registry_store import JSONFileVideoRegistry

registrar = VideoRegistrar(
    registry_store=JSONFileVideoRegistry("~/.video_analyzer/registry.json")
)
metadata = registrar.register_local_file("/path/to/local/video.mp4")
print(metadata["file_id"])
```
The registrar copies the video into the managed storage directory (or references it), saves metadata, and returns the new `file_id`. Swap the `registry_store` for a database-backed implementation later without touching the rest of the app.

You can also call the new gRPC helper directly:
```
grpcurl -plaintext -proto protos/video_analyzer.proto \
  -d '{"filePath":"/absolute/path/to/video.mp4","displayName":"My Clip","referenceOnly":false}' \
  localhost:50051 \
  video_analyzer.VideoAnalyzerService/RegisterLocalVideo
```
Set `referenceOnly` to true if you want the backend to use the original location instead of copying the file.

### End-to-end sanity tests
**Bash + grpcurl** (no Python client needed):
```
VIDEO_FILE_PATH=/path/to/video.mp4 \
  CHAT_MESSAGE='"Summarize the clip in one line"' \
  bash scripts/test_grpc_calls.sh
```

**Pure Python (uses generated stubs):**
```
uv run python scripts/test_grpc_calls.py \
  --file /path/to/video.mp4 \
  --chat-message "Summarize the clip in one line"
```
Both scripts register the file, call `SendChatMessage`, and fetch `GetChatHistory`.

### Storage layout
All user data lives under `~/Documents/VideoAnalyzer` (overridable via `VIDEO_ANALYZER_STORAGE_ROOT`):
- `videos/`: uploaded or registered video files
- `outputs/`: generated transcripts, detections, etc.
- `video_registry.json`: metadata index managed by `VideoRegistrar`

simple health cehck
```
nc -zv localhost 50051
```


## Generate Python Protobuf Types

**For new video_analyzer.proto:**
```bash
# Using uv (recommended)
uv run python -m grpc_tools.protoc \
  --python_out=protos/ \
  --grpc_python_out=protos/ \
  --proto_path=protos/ \
  protos/video_analyzer.proto

# Or using regular python
python -m grpc_tools.protoc \
  --python_out=protos/ \
  --grpc_python_out=protos/ \
  --proto_path=protos/ \
  protos/video_analyzer.proto
```

**After generation, fix the import in the generated gRPC file:**
```bash
# Change line 6 in protos/video_analyzer_pb2_grpc.py
# FROM: import video_analyzer_pb2 as video__analyzer__pb2
# TO:   from protos import video_analyzer_pb2 as video__analyzer__pb2
sed -i 's/^import video_analyzer_pb2/from protos import video_analyzer_pb2/' protos/video_analyzer_pb2_grpc.py
```

**Old api.proto (deprecated):**
```bash
# Only if you need to regenerate the old proto
uv run python -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --grpc_python_out=. \
  protos/video_analyzer.proto
```
