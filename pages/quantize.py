import asyncio
import re
import tempfile
import json
import os
import time
from nicegui import ui, run
from services import hf_service, quantization_service
import config

SCHEMES_INT = ['W4A16', 'W4A16_ASYM', 'W8A16', 'W4A8', 'W4AFP8']
SCHEMES_FP8 = ['FP8', 'FP8_DYNAMIC', 'FP8_BLOCK', 'NVFP4A16', 'NVFP4', 'MXFP4A16', 'MXFP4']
COMPRESSION_FORMATS = [
    'pack-quantized', 'int-quantized', 'float-quantized', 'dense',
    'sparse-bitmask', 'sparse-24-bitmask', 'naive-quantized',
    'marlin-24', 'mixed-precision', 'nvfp4-pack-quantized', 'mxfp4-pack-quantized',
]

CALIBRATION_DATASETS = {
    'ultrachat-200k': {
        'label': 'ultrachat-200k — Chat/instruct (llmcompressor default)',
        'split': 'train',
    },
    'open-platypus': {
        'label': 'open-platypus — Curated instruction data (Neural Magic)',
        'split': 'train',
    },
    'wikitext': {
        'label': 'wikitext — Wikipedia prose (classic GPTQ)',
        'split': 'train',
    },
    'c4': {
        'label': 'C4 — Web crawl (HF Transformers GPTQ default)',
        'split': 'train',
    },
    'gsm8k': {
        'label': 'gsm8k — Math word problems',
        'split': 'train',
    },
    'evolcodealpaca': {
        'label': 'evolcodealpaca — Code instruction pairs',
        'split': 'train',
    },
    'cnn-dailymail': {
        'label': 'cnn-dailymail — News summarization',
        'split': 'train',
    },
}

# Bits per param for size estimation
SCHEME_BITS = {
    'W4A16': 4, 'W4A16_ASYM': 4, 'W8A16': 8, 'W4A8': 4, 'W4AFP8': 4,
    'FP8': 8, 'FP8_DYNAMIC': 8, 'FP8_BLOCK': 8,
    'NVFP4A16': 4, 'NVFP4': 4, 'MXFP4A16': 4, 'MXFP4': 4,
}


def _info(text):
    """Inline info icon with tooltip."""
    with ui.icon('info_outline', size='xs').classes('text-grey cursor-pointer'):
        ui.tooltip(text).classes('max-w-xs')


def _get_model_dtype(model_id, model_info):
    """Read dtype from model's config.json."""
    try:
        if model_info.get('source') == 'local':
            cfg_path = os.path.join(model_info['path'], 'config.json')
        else:
            safe = model_id.replace('/', '--')
            import glob
            pattern = os.path.join(config.MODEL_CACHE_DIR, f'models--{safe}', 'snapshots', '*', 'config.json')
            matches = glob.glob(pattern)
            if not matches:
                return None
            cfg_path = matches[0]
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get('torch_dtype', cfg.get('dtype'))
    except Exception:
        return None


def content():
    """Quantize tab. Returns a refresh callable."""
    _model_info = {}
    _run_state = {'start_time': None}

    # Shared targets/ignore options as {name: label} dicts (populated on model select)
    _module_classes = {'Linear': 'Linear'}
    _module_ignore = {'lm_head': 'lm_head'}

    with ui.column().classes('w-full gap-2'):

        # ── Section 1: Model Selection ──
        ui.label('Model').classes('text-subtitle1 font-bold')
        with ui.row().classes('items-center gap-4 w-full'):
            model_select = ui.select([], label='Model (downloaded)', with_input=True).classes('w-96')
            model_dtype_label = ui.label('').classes('text-sm')

        # ── Section 2: Preprocessing ──
        preprocess_exp = ui.expansion('Preprocessing (Optional)', icon='tune').classes('w-full')
        with preprocess_exp:
            # SmoothQuant
            with ui.card().classes('w-full'):
                with ui.row().classes('items-center gap-1'):
                    sq_enable = ui.checkbox('SmoothQuant').classes('text-subtitle2 font-bold')
                    _info('Mathematically redistributes quantization difficulty from activations to weights. '
                          'Applied before quantization. Most useful with W8A8 schemes where both weights and '
                          'activations are quantized. Has minimal benefit for weight-only schemes like W4A16.')
                ui.label('Smooths activation outliers before quantization. Use with W8A8 schemes.').classes('text-xs text-grey')
                with ui.column().classes('w-full gap-2') as sq_fields:
                    with ui.row().classes('gap-4 items-center flex-wrap'):
                        sq_strength = ui.number('Smoothing strength', value=0.5, min=0.0, max=1.0, step=0.05).classes('w-40')
                        ui.label('Balance between weight and activation difficulty (0-1)').classes('text-xs text-grey self-center')
                        _info('Alpha parameter (0-1). At 0, all difficulty stays on activations. '
                              'At 1, all difficulty moves to weights. 0.5 is a balanced default.')
                    with ui.row().classes('gap-4 items-center flex-wrap'):
                        sq_cal_steps = ui.number('Calibration steps', value=0, min=0, step=1).classes('w-40')
                        ui.label('0 = use all calibration data').classes('text-xs text-grey self-center')
                        _info('Number of calibration samples to compute smoothing scales. 0 = use all available calibration data.')
                sq_fields.visible = False
                sq_enable.on_value_change(lambda e: setattr(sq_fields, 'visible', e.value))

            # SparseGPT
            with ui.card().classes('w-full mt-2'):
                with ui.row().classes('items-center gap-1'):
                    sp_enable = ui.checkbox('SparseGPT').classes('text-subtitle2 font-bold')
                    _info('One-shot pruning method that removes weights while minimizing output error. '
                          'Can be combined with quantization for compound compression '
                          '(e.g. 50% sparse + 4-bit = ~4x total compression).')
                ui.label('Prunes weights to create sparsity. Can combine with quantization.').classes('text-xs text-grey')
                with ui.column().classes('w-full gap-2') as sp_fields:
                    with ui.row().classes('gap-4 items-center flex-wrap'):
                        sp_sparsity = ui.number('Sparsity', value=0.5, min=0.0, max=1.0, step=0.05).classes('w-32')
                        ui.label('Target sparsity ratio (e.g. 0.5 = 50% zeros)').classes('text-xs text-grey self-center')
                        _info('Fraction of weights to zero out. 0.5 = remove 50% of weights. '
                              'Higher sparsity = more compression but more quality loss. '
                              '0.5 is a safe starting point, 0.7+ is aggressive.')
                    with ui.row().classes('gap-4 items-center flex-wrap'):
                        sp_mask = ui.select(['0:0', '2:4', '4:8'], value='0:0', label='Mask structure').classes('w-32')
                        ui.label('0:0=unstructured, 2:4=NVIDIA structured sparsity').classes('text-xs text-grey self-center')
                        _info('0:0 = unstructured (any weight can be pruned, flexible but no hardware speedup). '
                              '2:4 = NVIDIA structured sparsity (2 of every 4 weights pruned, '
                              'gets ~2x speedup on Ampere+ GPUs via sparse tensor cores). '
                              '4:8 = coarser structured pattern.')
                    with ui.row().classes('gap-4 items-center flex-wrap'):
                        sp_block_size = ui.number('Block size', value=128, min=1, step=1).classes('w-32')
                        ui.label('Columns processed per pass').classes('text-xs text-grey self-center')
                        _info('Number of weight columns processed together when solving the pruning optimization. '
                              '128 is the standard default. Larger values use more memory but can be slightly more accurate.')
                    with ui.row().classes('gap-4 items-center flex-wrap'):
                        sp_dampening = ui.number('Dampening frac', value=0.01, min=0.0, step=0.001, format='%.3f').classes('w-32')
                        ui.label('Hessian regularization').classes('text-xs text-grey self-center')
                        _info('Regularization added to the Hessian matrix diagonal to prevent numerical instability. '
                              '0.01 is the standard default. Increase if you see NaN errors.')
                    with ui.row().classes('items-center gap-1'):
                        sp_offload = ui.checkbox('Offload hessians to CPU to save VRAM')
                        _info('Move Hessian matrices to CPU RAM during computation. '
                              'Slower but saves GPU VRAM. Enable for very large models that don\'t fit in VRAM during quantization.')
                    with ui.row().classes('items-center gap-1'):
                        ui.label('Targets').classes('text-xs text-grey')
                        ui.label('— Layer types to prune').classes('text-xs text-grey')
                        _info('PyTorch module class names to prune. "Linear" targets all linear layers (recommended).')
                    sp_targets = ui.select(_module_classes, value=['Linear'], multiple=True).classes('w-full')
                    with ui.row().classes('items-center gap-1'):
                        ui.label('Ignore').classes('text-xs text-grey')
                        ui.label('— Layers to skip (overrides Targets)').classes('text-xs text-grey')
                        _info('Module instances to skip even if they match a Target class. '
                              '"lm_head" is the output projection — pruning it degrades quality significantly.')
                    sp_ignore = ui.select(_module_ignore, value=['lm_head'], multiple=True).classes('w-full')
                sp_fields.visible = False
                sp_enable.on_value_change(lambda e: setattr(sp_fields, 'visible', e.value))

        # ── Section 3: Quantization Method ──
        ui.label('Quantization Method').classes('text-subtitle1 font-bold mt-2')
        with ui.row().classes('items-center gap-1'):
            quant_method = ui.radio(['GPTQ', 'AWQ', 'FP8', 'AutoRound'], value='GPTQ').props('inline')
            _info('GPTQ: Calibration-based, widely supported, good INT4 quality. '
                  'AWQ: Activation-aware, often slightly better quality than GPTQ at same bits. '
                  'FP8: Fastest to run, minimal quality loss, native hardware support on Blackwell. '
                  'AutoRound: Intel\'s method, uses gradient descent for best rounding. Slowest but can produce highest quality.')

        # GPTQ fields
        with ui.card().classes('w-full') as gptq_card:
            with ui.row().classes('gap-4 items-center flex-wrap'):
                gptq_scheme = ui.select(SCHEMES_INT, value='W4A16', label='Scheme').classes('w-36')
                ui.label('Weight/activation bit width').classes('text-xs text-grey self-center')
                _info('W4A16: best compression. W4A16_ASYM: asymmetric variant, sometimes better. '
                      'W8A16: minimal quality loss, 2x compression. W4A8/W4AFP8: both weights and activations compressed.')
            with ui.row().classes('gap-4 items-center flex-wrap'):
                gptq_block_size = ui.number('Block size', value=128, min=1, step=1).classes('w-32')
                ui.label('Columns processed per pass').classes('text-xs text-grey self-center')
                _info('Columns processed per pass in the GPTQ algorithm. 128 is standard. '
                      'Smaller values use less memory but may be less accurate.')
            with ui.row().classes('gap-4 items-center flex-wrap'):
                gptq_dampening = ui.number('Dampening frac', value=0.01, min=0.0, step=0.001, format='%.3f').classes('w-32')
                ui.label('Hessian regularization').classes('text-xs text-grey self-center')
                _info('Hessian regularization. 0.01 is the standard default. '
                      'Increase if you see NaN errors during quantization.')
            with ui.row().classes('gap-4 items-center flex-wrap'):
                gptq_actorder = ui.select(['static', 'weight', 'dynamic'], value='static', label='Act order').classes('w-32')
                ui.label('Activation ordering strategy').classes('text-xs text-grey self-center')
                _info('"static": fixed order (fastest, good default). '
                      '"weight": order by weight magnitude (can improve quality). '
                      '"dynamic": recompute order per block (slowest, potentially best quality).')
            with ui.row().classes('items-center gap-1'):
                gptq_offload = ui.checkbox('Offload hessians to CPU to save VRAM')
                _info('Move Hessian matrices to CPU RAM. Slower but saves GPU VRAM for large models.')
            with ui.row().classes('items-center gap-1'):
                ui.label('Targets').classes('text-xs text-grey')
                ui.label('— Layer types to quantize').classes('text-xs text-grey')
                _info('PyTorch module class names to quantize. "Linear" targets all linear layers (recommended).')
            gptq_targets = ui.select(_module_classes, value=['Linear'], multiple=True).classes('w-full')
            with ui.row().classes('items-center gap-1'):
                ui.label('Ignore').classes('text-xs text-grey')
                ui.label('— Layers to skip (overrides Targets)').classes('text-xs text-grey')
                _info('Module instances to skip even if they match a Target class. '
                      '"lm_head" is the output layer — quantizing it hurts quality significantly.')
            gptq_ignore = ui.select(_module_ignore, value=['lm_head'], multiple=True).classes('w-full')

        # AWQ fields
        with ui.card().classes('w-full') as awq_card:
            with ui.row().classes('gap-4 items-center flex-wrap'):
                awq_scheme = ui.select(SCHEMES_INT, value='W4A16', label='Scheme').classes('w-36')
                ui.label('Weight/activation bit width').classes('text-xs text-grey self-center')
                _info('W4A16: best compression. W8A16: minimal quality loss. W4A8/W4AFP8: both compressed.')
            with ui.row().classes('items-center gap-1'):
                awq_duo = ui.checkbox('Duo scaling', value=True)
                ui.label('Use both activations & weights for scaling').classes('text-xs text-grey')
                _info('Use both activation and weight statistics to find optimal quantization scales. '
                      'Enabled by default. Disabling uses only activations, which is faster but lower quality.')
            with ui.row().classes('gap-4 items-center flex-wrap'):
                awq_ngrid = ui.number('Grid points', value=20, min=1, step=1).classes('w-32')
                ui.label('Grid search points (higher = better quality, slower)').classes('text-xs text-grey self-center')
                _info('Number of candidate scales tested during grid search. Higher = better quality but slower. '
                      '20 is the default. Try 40-100 for maximum quality on important models.')
            with ui.row().classes('items-center gap-1'):
                ui.label('Targets').classes('text-xs text-grey')
                ui.label('— Layer types to quantize').classes('text-xs text-grey')
                _info('PyTorch module class names to quantize. "Linear" targets all linear layers (recommended).')
            awq_targets = ui.select(_module_classes, value=['Linear'], multiple=True).classes('w-full')
            with ui.row().classes('items-center gap-1'):
                ui.label('Ignore').classes('text-xs text-grey')
                ui.label('— Layers to skip (overrides Targets)').classes('text-xs text-grey')
                _info('Module instances to skip even if they match a Target class.')
            awq_ignore = ui.select(_module_ignore, value=['lm_head'], multiple=True).classes('w-full')

        # FP8 fields
        with ui.card().classes('w-full') as fp8_card:
            with ui.row().classes('gap-4 items-center flex-wrap'):
                fp8_scheme = ui.select(SCHEMES_FP8, value='FP8_DYNAMIC', label='Scheme').classes('w-44')
                ui.label('FP8 variant').classes('text-xs text-grey self-center')
                _info('FP8: static scales (requires calibration). FP8_DYNAMIC: per-token dynamic (best default, no activation calibration). '
                      'FP8_BLOCK: block-wise 128x128. NVFP4/MXFP4: 4-bit float for Blackwell GPUs.')
            with ui.row().classes('items-center gap-1'):
                ui.label('Targets').classes('text-xs text-grey')
                ui.label('— Layer types to quantize').classes('text-xs text-grey')
                _info('PyTorch module class names to quantize. "Linear" targets all linear layers (recommended).')
            fp8_targets = ui.select(_module_classes, value=['Linear'], multiple=True).classes('w-full')
            with ui.row().classes('items-center gap-1'):
                ui.label('Ignore').classes('text-xs text-grey')
                ui.label('— Layers to skip (overrides Targets)').classes('text-xs text-grey')
                _info('Module instances to skip even if they match a Target class.')
            fp8_ignore = ui.select(_module_ignore, value=['lm_head'], multiple=True).classes('w-full')

        # AutoRound fields
        with ui.card().classes('w-full') as ar_card:
            with ui.row().classes('gap-4 items-center flex-wrap'):
                ar_scheme = ui.select(SCHEMES_INT, value='W4A16', label='Scheme').classes('w-36')
                ui.label('Weight/activation bit width').classes('text-xs text-grey self-center')
                _info('W4A16: best compression. W8A16: minimal quality loss. W4A8/W4AFP8: both compressed.')
            with ui.row().classes('gap-4 items-center flex-wrap'):
                ar_iters = ui.number('Iterations', value=200, min=1, step=10).classes('w-32')
                ui.label('Optimization iterations per block').classes('text-xs text-grey self-center')
                _info('Gradient descent iterations per transformer block. More iterations = better rounding decisions but slower. '
                      '200 is the default. 50-100 for quick runs, 500+ for maximum quality.')
            with ui.row().classes('gap-4 items-center flex-wrap'):
                ar_batch = ui.number('Batch size', value=8, min=1, step=1).classes('w-32')
                ui.label('Calibration batch size').classes('text-xs text-grey self-center')
                _info('Number of calibration samples per optimization step. '
                      'Larger batches are more stable but use more VRAM.')
            with ui.row().classes('gap-4 items-center flex-wrap'):
                ar_lr = ui.input('Learning rate', value='', placeholder='auto').classes('w-32')
                ui.label('Blank = auto').classes('text-xs text-grey self-center')
                _info('Learning rate for the rounding optimization. Leave blank for automatic selection based on model size.')
            with ui.row().classes('items-center gap-1'):
                ar_compile = ui.checkbox('Enable torch.compile', value=True)
                ui.label('Use torch.compile for speed').classes('text-xs text-grey')
                _info('Use torch.compile to JIT-compile the optimization loop. '
                      'Significantly faster but adds compilation time on first block.')
            with ui.row().classes('items-center gap-1'):
                ui.label('Targets').classes('text-xs text-grey')
                ui.label('— Layer types to quantize').classes('text-xs text-grey')
                _info('PyTorch module class names to quantize. "Linear" targets all linear layers (recommended).')
            ar_targets = ui.select(_module_classes, value=['Linear'], multiple=True).classes('w-full')
            with ui.row().classes('items-center gap-1'):
                ui.label('Ignore').classes('text-xs text-grey')
                ui.label('— Layers to skip (overrides Targets)').classes('text-xs text-grey')
                _info('Module instances to skip even if they match a Target class.')
            ar_ignore = ui.select(_module_ignore, value=['lm_head'], multiple=True).classes('w-full')

        method_cards = {'GPTQ': gptq_card, 'AWQ': awq_card, 'FP8': fp8_card, 'AutoRound': ar_card}

        def toggle_method():
            for name, card in method_cards.items():
                card.visible = (name == quant_method.value)

        quant_method.on_value_change(lambda _: toggle_method())
        toggle_method()

        # ── Section 4: Calibration ──
        cal_exp = ui.expansion('Calibration Settings', icon='dataset').classes('w-full')
        with cal_exp:
            with ui.card().classes('w-full'):
                ds_options = {k: v['label'] for k, v in CALIBRATION_DATASETS.items()}
                with ui.row().classes('gap-4 items-center flex-wrap'):
                    cal_dataset_select = ui.select(
                        ds_options, value='ultrachat-200k', label='Dataset'
                    ).classes('w-96')
                    _info('Calibration dataset. The model processes these samples to compute quantization statistics. '
                          'Use data that matches your model\'s intended use case for best results.')
                with ui.row().classes('gap-4 items-center flex-wrap'):
                    cal_dataset_custom = ui.input('Custom dataset (overrides above)', placeholder='e.g. my-org/my-dataset').classes('w-96')
                    _info('Enter a HuggingFace dataset ID to override the dropdown. Leave blank to use the selected dataset above.')
                with ui.row().classes('gap-4 items-center flex-wrap'):
                    cal_split = ui.input('Split', value='').classes('w-32')
                    ui.label('Auto-set from dropdown, or override').classes('text-xs text-grey self-center')
                    _info('Dataset split name (e.g. "train", "train_sft", "validation"). '
                          'Auto-filled when using a preset dataset. Override for custom datasets.')
                with ui.row().classes('gap-4 items-center flex-wrap'):
                    cal_samples = ui.number('Num calibration samples', value=512, min=1, step=64).classes('w-48')
                    _info('Number of dataset samples used. 512 is a good default. '
                          'More samples = better statistics but slower. 128 for quick tests, 1024+ for production.')
                    cal_seq_len = ui.number('Max seq length', value=384, min=1, step=64).classes('w-40')
                    _info('Maximum token length per calibration sample. Longer sequences capture more context '
                          'but use more VRAM. 384-512 is typical.')
                    cal_batch = ui.number('Batch size', value=8, min=1, step=1).classes('w-32')
                    _info('Calibration batch size. 1 is safest for VRAM. Increase if you have headroom.')

                def _on_dataset_select(_):
                    ds_id = cal_dataset_select.value
                    ds_info = CALIBRATION_DATASETS.get(ds_id, {})
                    cal_split.value = ds_info.get('split', '')

                cal_dataset_select.on_value_change(_on_dataset_select)
                _on_dataset_select(None)  # set initial split

        # ── Section 5: Output ──
        ui.label('Output').classes('text-subtitle1 font-bold mt-2')
        with ui.row().classes('gap-4 items-center flex-wrap w-full'):
            compression_select = ui.select(COMPRESSION_FORMATS, value='pack-quantized', label='Compression format').classes('w-48')
            ui.label('Output file format').classes('text-xs text-grey self-center')
            _info('pack-quantized: best for INT4. float-quantized: best for FP8. '
                  'marlin-24: fast GPU inference for 2:4 sparse. dense: no compression.')
        with ui.row().classes('gap-4 items-center w-full'):
            output_name = ui.input('Output name', value='').classes('w-96')
            output_dir_label = ui.label('').classes('text-xs text-grey')

        def _update_output_name():
            model = model_select.value
            if not model:
                return
            method = quant_method.value
            scheme_map = {'GPTQ': gptq_scheme, 'AWQ': awq_scheme, 'FP8': fp8_scheme, 'AutoRound': ar_scheme}
            scheme = scheme_map.get(method, gptq_scheme).value or ''
            suggested = quantization_service.suggest_output_name(model, method, scheme)
            output_name.value = suggested
            output_dir_label.set_text(os.path.join(config.LOCAL_MODELS_DIR, suggested))

        def _update_size_estimate():
            model_id = model_select.value
            info = _model_info.get(model_id, {})
            orig_bytes = info.get('size_bytes', 0)
            if not orig_bytes:
                size_label.set_text('')
                return
            orig_gb = orig_bytes / 1e9
            method = quant_method.value
            scheme_map = {'GPTQ': gptq_scheme, 'AWQ': awq_scheme, 'FP8': fp8_scheme, 'AutoRound': ar_scheme}
            scheme = scheme_map.get(method, gptq_scheme).value or ''
            bits = SCHEME_BITS.get(scheme, 16)
            # Estimate: original is typically 16-bit (2 bytes/param), quantized is bits/8 bytes/param
            est_bytes = orig_bytes * bits / 16
            est_gb = est_bytes / 1e9
            ratio = est_gb / orig_gb if orig_gb > 0 else 0
            size_label.set_text(f'Size: {orig_gb:.1f} GB -> ~{est_gb:.1f} GB ({ratio:.0%} of original)')

        # All targets/ignore selects to update when model changes
        _all_target_selects = [sp_targets, gptq_targets, awq_targets, fp8_targets, ar_targets]
        _all_ignore_selects = [sp_ignore, gptq_ignore, awq_ignore, fp8_ignore, ar_ignore]

        async def _on_model_change(_):
            _update_output_name()
            _update_size_estimate()
            model_id = model_select.value
            if not model_id:
                model_dtype_label.set_text('')
                return
            info = _model_info.get(model_id, {})
            dtype = _get_model_dtype(model_id, info)
            model_dtype_label.set_text(f'Format: {dtype}' if dtype else '')
            # Populate targets/ignore from model architecture
            model_path = info.get('path', model_id) if info.get('source') == 'local' else model_id
            modules = await run.io_bound(hf_service.get_model_modules, model_path)
            _module_classes.clear()
            _module_classes.update({c['name']: c['label'] for c in modules['classes']})
            _module_ignore.clear()
            _module_ignore.update({c['name']: c['label'] for c in modules['ignore']})
            class_names = list(_module_classes.keys())
            ignore_names = list(_module_ignore.keys())
            for sel in _all_target_selects:
                sel.options = dict(_module_classes)
                if not sel.value or sel.value == ['Linear']:
                    sel.value = ['Linear'] if 'Linear' in class_names else []
                sel.update()
            for sel in _all_ignore_selects:
                sel.options = dict(_module_ignore)
                if not sel.value or sel.value == ['lm_head']:
                    sel.value = ['lm_head'] if 'lm_head' in ignore_names else []
                sel.update()

        model_select.on_value_change(_on_model_change)
        quant_method.on_value_change(lambda _: (_update_output_name(), _update_size_estimate()))
        for s in [gptq_scheme, awq_scheme, fp8_scheme, ar_scheme]:
            s.on_value_change(lambda _: (_update_output_name(), _update_size_estimate()))
        output_name.on_value_change(lambda _: output_dir_label.set_text(
            os.path.join(config.LOCAL_MODELS_DIR, output_name.value) if output_name.value else ''))

        # ── Size Estimation ──
        size_label = ui.label('').classes('text-sm font-bold mt-2')

        # ── Section 6: Run Controls ──
        with ui.row().classes('gap-2 items-center mt-2'):
            run_btn = ui.button('Run Quantization', icon='play_arrow').props('color=positive')
            stop_btn = ui.button('Stop', icon='stop').props('color=negative')
            stop_btn.visible = False
            run_status = ui.label('').classes('text-sm')
            elapsed_label = ui.label('').classes('text-sm text-grey')

        progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
        progress_bar.visible = False
        progress_label = ui.label('').classes('text-xs text-grey')
        progress_label.visible = False

        quant_log = ui.log(max_lines=500).classes('w-full').style('height: 400px')
        quant_log.visible = False

    # ── Callbacks ──

    def _elapsed_str():
        if not _run_state['start_time']:
            return ''
        secs = int(time.time() - _run_state['start_time'])
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        if h:
            return f'{h}h {m}m {s}s'
        if m:
            return f'{m}m {s}s'
        return f'{s}s'

    def _tick_elapsed():
        if _run_state['start_time']:
            elapsed_label.set_text(_elapsed_str())

    ui.timer(1.0, _tick_elapsed)

    def _set_running(is_running):
        run_btn.set_enabled(not is_running)
        stop_btn.visible = is_running
        quant_log.visible = True
        progress_bar.visible = is_running
        progress_label.visible = is_running
        if is_running:
            progress_bar.value = 0
            progress_label.set_text('')
            _run_state['start_time'] = time.time()
            _run_state['step'] = 0
            _run_state['total_steps'] = 4
        else:
            _run_state['start_time'] = None

    def _update_progress(line):
        # Track our explicit STEP markers
        m = re.match(r'STEP (\d+)/(\d+):', line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            _run_state['step'] = step
            _run_state['total_steps'] = total
            progress_bar.value = (step - 1) / total
            progress_label.set_text(f'Step {step}/{total}: {line.split(":", 1)[1].strip()}')
            return
        # tqdm-style percentage
        m2 = re.search(r'(\d+)%\|', line)
        if m2:
            pct = int(m2.group(1))
            step = _run_state.get('step', 0)
            total = _run_state.get('total_steps', 4)
            base = (step - 1) / total if step > 0 else 0
            step_frac = pct / 100.0 / total
            progress_bar.value = base + step_frac
            progress_label.set_text(f'Step {step}/{total}: {pct}%')

    def _read_chunk(fd):
        try:
            return os.read(fd, 8192)
        except OSError:
            return b''

    def _client_alive():
        try:
            quant_log.client.check_existence()
            return True
        except Exception:
            return False

    async def _stream_proc(proc):
        fd = proc.stdout.fileno()
        buf = ''
        while proc.poll() is None:
            if not _client_alive():
                return
            raw = await run.io_bound(_read_chunk, fd)
            if not raw:
                await asyncio.sleep(0.1)
                continue
            buf += raw.decode('utf-8', errors='replace')
            parts = re.split(r'[\r\n]', buf)
            buf = parts[-1]
            for part in parts[:-1]:
                stripped = part.strip()
                if stripped:
                    quant_log.push(stripped)
                    _update_progress(stripped)
            await asyncio.sleep(0)
        if not _client_alive():
            return
        while True:
            raw = await run.io_bound(_read_chunk, fd)
            if not raw:
                break
            buf += raw.decode('utf-8', errors='replace')
        for part in re.split(r'[\r\n]', buf):
            stripped = part.strip()
            if stripped:
                quant_log.push(stripped)
                _update_progress(stripped)

    def _get_quant_params():
        method = quant_method.value
        if method == 'GPTQ':
            params = {
                'scheme': gptq_scheme.value,
                'block_size': int(gptq_block_size.value),
                'dampening_frac': float(gptq_dampening.value),
                'actorder': gptq_actorder.value,
                'offload_hessians': gptq_offload.value,
                'targets': list(gptq_targets.value or []),
                'ignore': list(gptq_ignore.value or []),
            }
        elif method == 'AWQ':
            params = {
                'scheme': awq_scheme.value,
                'duo_scaling': awq_duo.value,
                'n_grid': int(awq_ngrid.value),
                'targets': list(awq_targets.value or []),
                'ignore': list(awq_ignore.value or []),
            }
        elif method == 'FP8':
            params = {
                'scheme': fp8_scheme.value,
                'targets': list(fp8_targets.value or []),
                'ignore': list(fp8_ignore.value or []),
            }
        elif method == 'AutoRound':
            lr_val = ar_lr.value.strip()
            params = {
                'scheme': ar_scheme.value,
                'iters': int(ar_iters.value),
                'batch_size': int(ar_batch.value),
                'lr': float(lr_val) if lr_val else None,
                'enable_torch_compile': ar_compile.value,
                'targets': list(ar_targets.value or []),
                'ignore': list(ar_ignore.value or []),
            }
        return params

    def _get_smoothquant_config():
        if not sq_enable.value:
            return None
        return {
            'enabled': True,
            'smoothing_strength': float(sq_strength.value),
            'num_calibration_steps': int(sq_cal_steps.value) if sq_cal_steps.value and int(sq_cal_steps.value) > 0 else None,
        }

    def _get_sparsegpt_config():
        if not sp_enable.value:
            return None
        return {
            'enabled': True,
            'sparsity': float(sp_sparsity.value),
            'mask_structure': sp_mask.value,
            'block_size': int(sp_block_size.value),
            'dampening_frac': float(sp_dampening.value),
            'offload_hessians': sp_offload.value,
            'targets': list(sp_targets.value or []),
            'ignore': list(sp_ignore.value or []),
        }

    async def run_quantization():
        model_id = model_select.value
        if not model_id:
            ui.notify('Select a model first', type='warning')
            return
        name = output_name.value.strip()
        if not name:
            ui.notify('Enter an output name', type='warning')
            return

        # Resolve model path
        info = _model_info.get(model_id, {})
        model = info.get('path', model_id) if info.get('source') == 'local' else model_id

        # Check if output already exists
        out_dir = os.path.join(config.LOCAL_MODELS_DIR, name)
        if os.path.exists(out_dir):
            with ui.dialog() as confirm, ui.card():
                ui.label(f'"{name}" already exists. Overwrite?')
                with ui.row():
                    ui.button('Overwrite', on_click=lambda: confirm.submit(True)).props('color=negative')
                    ui.button('Cancel', on_click=lambda: confirm.submit(False)).props('flat')
            result = await confirm
            if not result:
                return

        cfg = quantization_service.build_config(
            model=model,
            model_id=model_id,
            smoothquant=_get_smoothquant_config(),
            sparsegpt=_get_sparsegpt_config(),
            quant_method=quant_method.value.lower(),
            quant_params=_get_quant_params(),
            calibration={
                'dataset': cal_dataset_custom.value.strip() or cal_dataset_select.value,
                'split': cal_split.value.strip() or 'train',
                'num_calibration_samples': int(cal_samples.value),
                'max_seq_length': int(cal_seq_len.value),
                'batch_size': int(cal_batch.value),
            },
            output_name=name,
            compression_format=compression_select.value,
        )

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(cfg, tmp, indent=2)
        tmp.close()

        _set_running(True)
        quant_log.clear()
        run_status.set_text('Starting quantization...')

        try:
            proc = await run.io_bound(quantization_service.run_quantization, tmp.name)
            await _stream_proc(proc)
            if _client_alive():
                elapsed = _elapsed_str()
                if proc.returncode == 0:
                    progress_bar.value = 1.0
                    progress_label.set_text('Complete')
                    run_status.set_text(f'Quantization complete ({elapsed})')
                    ui.notify('Quantization complete', type='positive')
                else:
                    run_status.set_text(f'Quantization failed (exit code {proc.returncode}, {elapsed})')
                    ui.notify(f'Quantization failed (exit code {proc.returncode})', type='negative')
        except Exception as e:
            if _client_alive():
                run_status.set_text(f'Error: {e}')
                ui.notify(f'Error: {e}', type='negative')
        finally:
            if _client_alive():
                _set_running(False)
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def on_stop():
        quantization_service.stop_quantization()
        elapsed = _elapsed_str()
        run_status.set_text(f'Quantization stopped ({elapsed})')
        ui.notify('Quantization stopped', type='info')
        _set_running(False)

    async def refresh_models():
        cached = await run.io_bound(hf_service.list_cached_models)
        _model_info.clear()
        for m in cached:
            model_path = m.get('path', m['id']) if m.get('source') == 'local' else m['id']
            m['quant_status'] = await run.io_bound(hf_service.get_quantization_status, model_path)
            _model_info[m['id']] = m
        options = {}
        for m in cached:
            label = m['id']
            if m.get('quant_status'):
                label += f'  ⚠ QUANTIZED ({m["quant_status"]})'
            options[m['id']] = label
        model_select.options = options
        if len(options) == 1:
            model_select.value = list(options.keys())[0]
        model_select.update()

    run_btn.on_click(run_quantization)
    stop_btn.on_click(on_stop)
    ui.timer(0.1, refresh_models, once=True)

    return refresh_models
