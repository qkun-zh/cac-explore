import os

def setup_hf_env():
    os.environ.setdefault("HF_HOME", "/data/asset/hf")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def hf_token():
    for p in ("/tmp/hf_token.txt", os.path.expanduser("~/.cache/huggingface/token")):
        if os.path.exists(p):
            return open(p).read().strip()
    return None
