import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is listed in requirements.txt, but keep direct script use
    # tolerant if only environment variables are exported.
    pass


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/opt/hpml_project/models")
LOCAL_MODELS_DIR = os.environ.get("LOCAL_MODELS_DIR", os.path.join(MODEL_CACHE_DIR, "local"))
os.environ["HF_HUB_CACHE"] = MODEL_CACHE_DIR
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = _env_int("APP_PORT", 8080)
APP_HOSTNAME = os.environ.get("APP_HOSTNAME", "precision-wsl")
STORAGE_SECRET = os.environ.get("STORAGE_SECRET", "hpml-model-manager-secret")

VLLM_PORT_START = _env_int("VLLM_PORT_START", 8001)

DEFAULT_GPU_MEM_UTIL = _env_float("DEFAULT_GPU_MEM_UTIL", 0.90)
DEFAULT_DTYPE = os.environ.get("DEFAULT_DTYPE", "auto")
DEFAULT_SEARCH_LIMIT = _env_int("DEFAULT_SEARCH_LIMIT", 500)
