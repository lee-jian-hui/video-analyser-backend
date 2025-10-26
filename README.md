## ENVIRONMENT
- WSL Linux Ubuntu 22.04
- Python 3.10.12


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