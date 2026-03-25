"""Wrapper script for llmcompressor quantization. Invoked as a subprocess.
Usage: python run_quantize.py /path/to/config.json
"""
import sys
import json
import os
import getpass
from datetime import datetime


def build_recipe(config):
    recipe = []

    # Preprocessing: SmoothQuant
    sq = config.get('smoothquant')
    if sq and sq.get('enabled'):
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        params = {k: v for k, v in sq.items() if k != 'enabled' and v is not None}
        recipe.append(SmoothQuantModifier(**params))

    # Preprocessing: SparseGPT
    sp = config.get('sparsegpt')
    if sp and sp.get('enabled'):
        from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
        params = {k: v for k, v in sp.items() if k != 'enabled' and v is not None}
        recipe.append(SparseGPTModifier(**params))

    # Quantization method
    method = config['quant_method']
    qparams = dict(config['quant_params'])
    # Remove None values
    qparams = {k: v for k, v in qparams.items() if v is not None}

    if method == 'gptq':
        from llmcompressor.modifiers.quantization import GPTQModifier
        recipe.append(GPTQModifier(**qparams))
    elif method == 'awq':
        from llmcompressor.modifiers.awq import AWQModifier
        recipe.append(AWQModifier(**qparams))
    elif method == 'fp8':
        from llmcompressor.modifiers.quantization import QuantizationModifier
        recipe.append(QuantizationModifier(**qparams))
    elif method == 'autoround':
        from llmcompressor.modifiers.autoround import AutoRoundModifier
        recipe.append(AutoRoundModifier(**qparams))

    return recipe


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    recipe = build_recipe(config)
    cal = config.get('calibration', {})

    print(f"Model: {config['model']}")
    print(f"Method: {config['quant_method']}")
    print(f"Output: {config['output_dir']}")
    print(f"Recipe: {[type(m).__name__ for m in recipe]}")
    print(flush=True)

    # Load model onto GPU
    print("STEP 1/4: Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(config['model'], dtype="auto").to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    print("STEP 2/4: Downloading and preparing calibration data (this may take a while on first run)...")
    sys.stdout.flush()

    from llmcompressor import oneshot

    oneshot_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=config['output_dir'],
        save_compressed=config.get('save_compressed', True),
    )

    # Calibration settings
    dataset = cal.get('dataset')
    if dataset:
        oneshot_kwargs['dataset'] = dataset
    # Always pass splits if provided
    split = cal.get('split')
    if split:
        oneshot_kwargs['splits'] = split
    if cal.get('num_calibration_samples'):
        oneshot_kwargs['num_calibration_samples'] = cal['num_calibration_samples']
    if cal.get('max_seq_length'):
        oneshot_kwargs['max_seq_length'] = cal['max_seq_length']
    if cal.get('batch_size'):
        oneshot_kwargs['batch_size'] = cal['batch_size']

    # Parallelize dataset preprocessing across physical CPU cores
    num_workers = 24
    oneshot_kwargs['preprocessing_num_workers'] = num_workers
    print(f"  Using {num_workers} workers for dataset preprocessing")

    print("STEP 3/4: Quantizing...")
    oneshot(**oneshot_kwargs)

    print("STEP 4/4: Saving model...")
    # Write metadata
    meta = {
        'source_model': config.get('model_id', config['model']),
        'method': config['quant_method'],
        'scheme': config.get('quant_params', {}).get('scheme', ''),
        'preprocess': [],
        'created': datetime.now().isoformat(),
        'created_by': getpass.getuser(),
    }
    if (config.get('smoothquant') or {}).get('enabled'):
        meta['preprocess'].append('smoothquant')
    if (config.get('sparsegpt') or {}).get('enabled'):
        meta['preprocess'].append('sparsegpt')

    meta_path = os.path.join(config['output_dir'], '.model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"QUANTIZATION_COMPLETE: {config['output_dir']}")


if __name__ == '__main__':
    main()
