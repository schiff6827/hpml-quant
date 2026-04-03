import os

MODEL_CACHE_DIR = "/opt/hpml_project/models"
LOCAL_MODELS_DIR = os.path.join(MODEL_CACHE_DIR, "local")
os.environ["HF_HUB_CACHE"] = MODEL_CACHE_DIR
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

APP_HOST = "0.0.0.0"
APP_PORT = 8080
APP_HOSTNAME = "precision-wsl"
STORAGE_SECRET = "hpml-model-manager-secret"

VLLM_PORT_START = 8001

DEFAULT_GPU_MEM_UTIL = 0.90
DEFAULT_DTYPE = "auto"
DEFAULT_SEARCH_LIMIT = 500
