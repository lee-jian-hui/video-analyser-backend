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