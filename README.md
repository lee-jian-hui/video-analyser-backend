## ENVIRONMENT
- WSL Linux Ubuntu 22.04
- Python 3.10.12


## PRE-REQ
- have `pip` or `uv` to install dependencies
- `pip install uv` to get uv which is a python package manager
- `cp sample_env.txt`


## LOCAL SETUP
```
# use a dependency manager like conda 
# i use uv here
uv sync
python download.py
python main.py
```