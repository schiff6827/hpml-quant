import subprocess
import signal
import sys
import os
import config

_active_proc = None

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')


def run_quantization(config_path):
    global _active_proc
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, 'run_quantize.py'),
        config_path,
    ]
    env = {**os.environ, "HF_HUB_CACHE": config.MODEL_CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    _active_proc = proc
    return proc


def stop_quantization():
    global _active_proc
    if _active_proc and _active_proc.poll() is None:
        _active_proc.send_signal(signal.SIGTERM)
        try:
            _active_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _active_proc.kill()
    _active_proc = None


def is_running():
    return _active_proc is not None and _active_proc.poll() is None


def suggest_output_name(model_id, method, scheme):
    name = model_id.split('/')[-1] if '/' in model_id else model_id
    parts = [name, method.upper()]
    if scheme:
        parts.append(scheme)
    return '-'.join(parts)


def build_config(model, model_id, smoothquant, sparsegpt, quant_method, quant_params,
                 calibration, output_name, compression_format):
    output_dir = os.path.join(config.LOCAL_MODELS_DIR, output_name)
    return {
        'model': model,
        'model_id': model_id,
        'smoothquant': smoothquant,
        'sparsegpt': sparsegpt,
        'quant_method': quant_method,
        'quant_params': quant_params,
        'calibration': calibration,
        'output_dir': output_dir,
        'output_name': output_name,
        'compression_format': compression_format,
        'save_compressed': True,
    }


def build_bnb_config(model, model_id, bnb_params, output_name):
    output_dir = os.path.join(config.LOCAL_MODELS_DIR, output_name)
    return {
        'model': model,
        'model_id': model_id,
        'quant_method': 'bitsandbytes',
        'bnb_params': bnb_params,
        'output_dir': output_dir,
        'output_name': output_name,
    }
