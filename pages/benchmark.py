import asyncio
import re
import tempfile
import os
from collections import deque
from nicegui import ui, run
from services import vllm_service, benchmark_service, metrics_service, queue_service

_PROFILE_MAX_POINTS = 240


def content():
    """Benchmark tab. Returns a refresh callable."""
    with ui.column().classes('w-full gap-2'):
        # --- Server selector + run name ---
        with ui.row().classes('items-center gap-4 w-full'):
            server_select = ui.select([], label='Server', with_input=True).classes('w-64')
            refresh_btn = ui.button('', icon='refresh').props('flat dense')
            run_name_input = ui.input('Run name', placeholder='e.g. Qwen3-0.6B-BF16').classes('w-64')

        # --- Saved scripts ---
        with ui.row().classes('items-center gap-2 w-full'):
            script_select = ui.select([], label='Script').classes('w-64')
            script_load_btn = ui.button('Load', icon='folder_open').props('flat dense')
            script_name_input = ui.input('Save as', placeholder='script name').classes('w-48')
            script_save_btn = ui.button('Save', icon='save').props('flat dense')
            script_delete_btn = ui.button('Delete', icon='delete').props('flat dense color=negative')

        # --- Preset buttons ---
        ui.label('Presets').classes('text-subtitle2 font-bold mt-2')
        with ui.row().classes('gap-2'):
            preset_quick_perf = ui.button('Quick Perf').props('outline')
            preset_full_perf = ui.button('Full Perf').props('outline')
            preset_quick_qual = ui.button('Quick Quality').props('outline')
            preset_full_qual = ui.button('Full Quality').props('outline')
            preset_max_ctx = ui.button('Max Context').props('outline')
            preset_custom = ui.button('Custom').props('outline')
        preset_btns = [preset_quick_perf, preset_full_perf, preset_quick_qual,
                       preset_full_qual, preset_max_ctx, preset_custom]

        def _highlight_preset(active):
            for btn in preset_btns:
                if btn is active:
                    btn.props(remove='outline')
                    btn.props('color=primary')
                else:
                    btn.props(remove='color=primary')
                    btn.props('outline')

        # --- Advanced panel ---
        advanced = ui.expansion('Advanced Settings', icon='tune').classes('w-full')
        with advanced:
            with ui.card().classes('w-full'):
                perf_enable = ui.checkbox('Performance (vllm bench serve)', value=True).classes('text-subtitle2 font-bold')
                with ui.row().classes('gap-4 items-end flex-wrap') as perf_fields:
                    dataset_select = ui.select(['random', 'sharegpt', 'sonnet'], value='random', label='Dataset').classes('w-36')
                    num_prompts_input = ui.number('Num prompts', value=500, min=1, step=100).classes('w-32')
                    request_rate_input = ui.input('Request rate', value='inf').classes('w-32')
                    max_concurrency_input = ui.number('Max concurrency', value=64, min=1, step=8).classes('w-32')
                with ui.row().classes('gap-4 items-end') as random_row:
                    random_input_len = ui.number('Input length', value=512, min=1).classes('w-32')
                    random_output_len = ui.number('Output length', value=256, min=1).classes('w-32')

                def toggle_random_fields():
                    is_random = dataset_select.value == 'random'
                    random_row.visible = is_random

                dataset_select.on_value_change(lambda _: toggle_random_fields())
                toggle_random_fields()
                perf_enable.on_value_change(lambda e: setattr(perf_fields, 'visible', e.value) or (setattr(random_row, 'visible', e.value and dataset_select.value == 'random')))

            with ui.card().classes('w-full mt-2'):
                qual_enable = ui.checkbox('Quality (lm-eval)', value=False).classes('text-subtitle2 font-bold')
                all_tasks = ['mmlu', 'hellaswag', 'arc_challenge', 'arc_easy',
                             'truthfulqa_mc1', 'truthfulqa_mc2', 'gsm8k']
                task_checks = {}
                with ui.column().classes('w-full gap-2') as qual_fields:
                    with ui.row().classes('gap-4 flex-wrap'):
                        for t in all_tasks:
                            task_checks[t] = ui.checkbox(t, value=False)
                    with ui.row().classes('gap-4 items-end'):
                        num_fewshot_input = ui.number('Num few-shot', value=5, min=0).classes('w-32')
                        num_concurrent_input = ui.number('Num concurrent', value=16, min=1).classes('w-32')
                        limit_input = ui.number('Limit (0=all)', value=0, min=0).classes('w-32')
                qual_fields.visible = False
                qual_enable.on_value_change(lambda e: setattr(qual_fields, 'visible', e.value))

            with ui.card().classes('w-full mt-2'):
                ctx_enable = ui.checkbox('Max Context Sweep', value=False).classes('text-subtitle2 font-bold')
                with ui.row().classes('gap-4 items-end') as ctx_fields:
                    ctx_upper_input = ui.number('Upper bound', value=131072, min=1024, step=1024).classes('w-36')
                    ctx_step_input = ui.number('Step', value=4096, min=256, step=256).classes('w-32')
                ctx_fields.visible = False
                ctx_enable.on_value_change(lambda e: setattr(ctx_fields, 'visible', e.value))

        # --- Run controls ---
        with ui.row().classes('gap-2 items-center mt-2'):
            run_btn = ui.button('Run Benchmark', icon='play_arrow').props('color=positive')
            queue_add_btn = ui.button('Add to Queue', icon='playlist_add').props('color=secondary')
            stop_btn = ui.button('Stop', icon='stop').props('color=negative')
            stop_btn.visible = False
            run_status = ui.label('').classes('text-sm')
            queue_count_label = ui.label('').classes('text-xs text-grey')

        # --- Progress bar ---
        progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
        progress_bar.visible = False
        progress_label = ui.label('').classes('text-xs text-grey')
        progress_label.visible = False

        # --- Progress log ---
        bench_log = ui.log(max_lines=500).classes('w-full').style('height: 400px')
        bench_log.visible = False

        # --- Results section ---
        ui.separator().classes('mt-2')
        ui.label('Results').classes('text-subtitle1 font-bold')

        perf_table = ui.table(
            columns=[
                {'name': 'metric', 'label': 'Metric', 'field': 'metric', 'align': 'left'},
                {'name': 'value', 'label': 'Value', 'field': 'value', 'align': 'right'},
            ],
            rows=[],
        ).classes('w-full')
        perf_table.visible = False

        qual_table = ui.table(
            columns=[
                {'name': 'task', 'label': 'Task', 'field': 'task', 'align': 'left'},
                {'name': 'acc', 'label': 'Accuracy', 'field': 'acc', 'align': 'right'},
                {'name': 'stderr', 'label': 'Stderr', 'field': 'stderr', 'align': 'right'},
            ],
            rows=[],
        ).classes('w-full')
        qual_table.visible = False

        ctx_headline = ui.label('').classes('text-h6')
        ctx_headline.visible = False
        ctx_table = ui.table(
            columns=[
                {'name': 'n', 'label': 'Prompt tokens', 'field': 'n', 'align': 'right'},
                {'name': 'success', 'label': 'Success', 'field': 'success', 'align': 'center'},
                {'name': 'ttft_ms', 'label': 'TTFT (ms)', 'field': 'ttft_ms', 'align': 'right'},
                {'name': 'kv_cache_pct_at_end', 'label': 'KV %', 'field': 'kv_cache_pct_at_end', 'align': 'right'},
                {'name': 'error', 'label': 'Error', 'field': 'error', 'align': 'left'},
            ],
            rows=[],
        ).classes('w-full')
        ctx_table.visible = False

        # --- Previous results ---
        ui.separator()
        ui.label('Previous Results').classes('text-subtitle2 font-bold')
        with ui.row().classes('gap-2 items-end'):
            prev_select = ui.select([], label='Load result').classes('w-80')
            load_btn = ui.button('Load', icon='folder_open').props('flat')
            refresh_results_btn = ui.button('', icon='refresh').props('flat dense')

        prev_perf_table = ui.table(
            columns=[
                {'name': 'metric', 'label': 'Metric', 'field': 'metric', 'align': 'left'},
                {'name': 'value', 'label': 'Value', 'field': 'value', 'align': 'right'},
            ],
            rows=[],
        ).classes('w-full')
        prev_perf_table.visible = False

        prev_qual_table = ui.table(
            columns=[
                {'name': 'task', 'label': 'Task', 'field': 'task', 'align': 'left'},
                {'name': 'acc', 'label': 'Accuracy', 'field': 'acc', 'align': 'right'},
                {'name': 'stderr', 'label': 'Stderr', 'field': 'stderr', 'align': 'right'},
            ],
            rows=[],
        ).classes('w-full')
        prev_qual_table.visible = False

        # --- Live Profiling ---
        ui.separator()
        profile_exp = ui.expansion('Profiling', icon='speed').classes('w-full')
        with profile_exp:
            with ui.row().classes('items-center gap-4 w-full'):
                profile_status = ui.label('Idle').classes('text-sm text-grey')
                profile_csv_label = ui.label('').classes('text-xs text-grey')
            with ui.row().classes('gap-3 flex-wrap'):
                with ui.card().classes('p-2'):
                    ui.label('Peak VRAM').classes('text-caption')
                    profile_peak_vram = ui.label('--').classes('text-subtitle1')
                with ui.card().classes('p-2'):
                    ui.label('Peak GPU util').classes('text-caption')
                    profile_peak_util = ui.label('--').classes('text-subtitle1')
                with ui.card().classes('p-2'):
                    ui.label('Peak power').classes('text-caption')
                    profile_peak_power = ui.label('--').classes('text-subtitle1')
                with ui.card().classes('p-2'):
                    ui.label('Peak CPU RSS').classes('text-caption')
                    profile_peak_rss = ui.label('--').classes('text-subtitle1')
            with ui.row().classes('w-full gap-2'):
                profile_gpu_chart = ui.echart({
                    'title': {'text': 'GPU util / temp / power', 'textStyle': {'fontSize': 13}},
                    'tooltip': {'trigger': 'axis'},
                    'legend': {'data': ['Util %', 'Temp C', 'Power W'], 'bottom': 0},
                    'xAxis': {'type': 'category', 'data': []},
                    'yAxis': [
                        {'type': 'value', 'min': 0, 'max': 100, 'name': '% / C', 'position': 'left'},
                        {'type': 'value', 'name': 'W', 'position': 'right'},
                    ],
                    'series': [
                        {'name': 'Util %', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 0},
                        {'name': 'Temp C', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 0},
                        {'name': 'Power W', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 1},
                    ],
                    'animation': False,
                }).classes('w-1/2 h-64')
                profile_mem_chart = ui.echart({
                    'title': {'text': 'VRAM / KV cache', 'textStyle': {'fontSize': 13}},
                    'tooltip': {'trigger': 'axis'},
                    'legend': {'data': ['VRAM GB', 'KV %'], 'bottom': 0},
                    'xAxis': {'type': 'category', 'data': []},
                    'yAxis': [
                        {'type': 'value', 'name': 'GB', 'position': 'left'},
                        {'type': 'value', 'min': 0, 'max': 100, 'name': '%', 'position': 'right'},
                    ],
                    'series': [
                        {'name': 'VRAM GB', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'areaStyle': {}, 'yAxisIndex': 0},
                        {'name': 'KV %', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 1},
                    ],
                    'animation': False,
                }).classes('w-1/2 h-64')
            with ui.row().classes('w-full gap-2'):
                profile_tput_chart = ui.echart({
                    'title': {'text': 'Throughput & requests', 'textStyle': {'fontSize': 13}},
                    'tooltip': {'trigger': 'axis'},
                    'legend': {'data': ['tok/s', 'Running', 'Waiting'], 'bottom': 0},
                    'xAxis': {'type': 'category', 'data': []},
                    'yAxis': [
                        {'type': 'value', 'name': 'tok/s', 'position': 'left'},
                        {'type': 'value', 'name': 'reqs', 'position': 'right'},
                    ],
                    'series': [
                        {'name': 'tok/s', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 0},
                        {'name': 'Running', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 1},
                        {'name': 'Waiting', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 1},
                    ],
                    'animation': False,
                }).classes('w-full h-64')

        # --- Pareto frontier ---
        ui.separator()
        pareto_exp = ui.expansion('Pareto Frontier', icon='scatter_plot').classes('w-full')
        with pareto_exp:
            _axis_options = {
                'quality_score': 'Quality (task accuracy)',
                'throughput_tps': 'Throughput (tok/s)',
                'prefill_tps': 'Prefill throughput (tok/s)',
                'parameters_b': 'Parameters (B)',
                'size_gb': 'Model size (GB)',
                'peak_vram_gb': 'Peak VRAM (GB)',
                'avg_gpu_power_w': 'Avg GPU power (W)',
            }
            with ui.row().classes('items-end gap-2 w-full flex-wrap'):
                pareto_yaxis_select = ui.select(
                    _axis_options, value='quality_score', label='Y axis',
                ).classes('w-48')
                pareto_xaxis_select = ui.select(
                    _axis_options, value='throughput_tps', label='X axis',
                ).classes('w-48')
                pareto_task_select = ui.select([], label='Quality task').classes('w-40')
                pareto_colorby_select = ui.select(
                    {'quantization': 'Quantization', 'model_family': 'Model family'},
                    value='quantization', label='Color by',
                ).classes('w-40')
            with ui.row().classes('items-end gap-2 w-full flex-wrap'):
                pareto_run_filter = ui.select(
                    [], label='Runs (empty = all)', multiple=True,
                ).props('use-chips').classes('w-96')
                pareto_quant_filter = ui.select(
                    [], label='Quant filter (empty = all)', multiple=True,
                ).props('use-chips').classes('w-56')
                pareto_refresh_btn = ui.button('Refresh', icon='refresh').props('flat dense')
                pareto_backfill_btn = ui.button('Backfill metadata on old runs', icon='build').props('flat dense')
            pareto_empty_label = ui.label('').classes('text-caption text-grey')
            pareto_chart = ui.echart({
                'title': {'text': '', 'textStyle': {'fontSize': 13}},
                'tooltip': {'trigger': 'item'},
                'legend': {'bottom': 0},
                'xAxis': {'type': 'value', 'name': 'Output tokens/sec'},
                'yAxis': {'type': 'value', 'name': 'Quality', 'min': 0, 'max': 1},
                'series': [],
                'animation': False,
            }).classes('w-full h-80')

        # --- Compare ---
        ui.separator()
        ui.label('Compare Results').classes('text-subtitle2 font-bold')
        with ui.row().classes('gap-2 items-end'):
            cmp_a = ui.select([], label='Run A').classes('w-64')
            cmp_b = ui.select([], label='Run B').classes('w-64')
            cmp_btn = ui.button('Compare', icon='compare_arrows').props('flat')

        cmp_table = ui.table(
            columns=[
                {'name': 'metric', 'label': 'Metric', 'field': 'metric', 'align': 'left'},
                {'name': 'a', 'label': 'Run A', 'field': 'a', 'align': 'right'},
                {'name': 'b', 'label': 'Run B', 'field': 'b', 'align': 'right'},
                {'name': 'delta', 'label': 'Delta', 'field': 'delta', 'align': 'right'},
            ],
            rows=[],
        ).classes('w-full')
        cmp_table.visible = False

    # ---- Callbacks ----

    _saved_results_cache = []
    _profile_state = {
        'run_name': None,
        'csv_path': None,
        'timestamps': deque(maxlen=_PROFILE_MAX_POINTS),
        'util': deque(maxlen=_PROFILE_MAX_POINTS),
        'temp': deque(maxlen=_PROFILE_MAX_POINTS),
        'power': deque(maxlen=_PROFILE_MAX_POINTS),
        'vram_gb': deque(maxlen=_PROFILE_MAX_POINTS),
        'kv_pct': deque(maxlen=_PROFILE_MAX_POINTS),
        'tps': deque(maxlen=_PROFILE_MAX_POINTS),
        'running': deque(maxlen=_PROFILE_MAX_POINTS),
        'waiting': deque(maxlen=_PROFILE_MAX_POINTS),
    }

    def _reset_profile_buffers(run_name, csv_path):
        _profile_state['run_name'] = run_name
        _profile_state['csv_path'] = csv_path
        for k in ('timestamps', 'util', 'temp', 'power', 'vram_gb',
                  'kv_pct', 'tps', 'running', 'waiting'):
            _profile_state[k].clear()
        profile_csv_label.set_text(f'CSV: {csv_path}' if csv_path else '')
        profile_peak_vram.set_text('--')
        profile_peak_util.set_text('--')
        profile_peak_power.set_text('--')
        profile_peak_rss.set_text('--')

    def _apply_profile_charts():
        xs = list(_profile_state['timestamps'])
        profile_gpu_chart.options['xAxis']['data'] = xs
        profile_gpu_chart.options['series'][0]['data'] = list(_profile_state['util'])
        profile_gpu_chart.options['series'][1]['data'] = list(_profile_state['temp'])
        profile_gpu_chart.options['series'][2]['data'] = list(_profile_state['power'])
        profile_gpu_chart.update()
        profile_mem_chart.options['xAxis']['data'] = xs
        profile_mem_chart.options['series'][0]['data'] = list(_profile_state['vram_gb'])
        profile_mem_chart.options['series'][1]['data'] = list(_profile_state['kv_pct'])
        profile_mem_chart.update()
        profile_tput_chart.options['xAxis']['data'] = xs
        profile_tput_chart.options['series'][0]['data'] = list(_profile_state['tps'])
        profile_tput_chart.options['series'][1]['data'] = list(_profile_state['running'])
        profile_tput_chart.options['series'][2]['data'] = list(_profile_state['waiting'])
        profile_tput_chart.update()

    def poll_profile():
        rname = _profile_state['run_name']
        if not rname or not metrics_service.is_run_recording(rname):
            return
        latest = metrics_service.get_run_latest(rname)
        if not latest:
            return
        ts = (latest.get('timestamp') or '')[-8:]
        _profile_state['timestamps'].append(ts)
        _profile_state['util'].append(round(float(latest.get('gpu_util_pct') or 0), 1))
        _profile_state['temp'].append(round(float(latest.get('gpu_temp_c') or 0), 1))
        _profile_state['power'].append(round(float(latest.get('gpu_power_w') or 0), 1))
        vram_mb = float(latest.get('gpu_mem_used_mb') or 0)
        _profile_state['vram_gb'].append(round(vram_mb / 1024.0, 2))
        _profile_state['kv_pct'].append(round(float(latest.get('kv_cache_pct') or 0), 1))
        # gen_tokens_per_sec is derived live in fetch_vllm_metrics but not stored
        # in CSV — fall back to 0.
        _profile_state['tps'].append(round(float(latest.get('gen_tokens_per_sec') or 0), 1))
        _profile_state['running'].append(float(latest.get('requests_running') or 0))
        _profile_state['waiting'].append(float(latest.get('requests_waiting') or 0))
        _apply_profile_charts()
        peaks = metrics_service.get_run_peaks(rname)
        if peaks:
            pv = peaks.get('gpu_mem_peak_mb') or 0
            pp = peaks.get('gpu_power_peak_w') or 0
            pr = peaks.get('rss_peak_mb') or 0
            profile_peak_vram.set_text(f'{pv/1024:.1f} GB' if pv else '--')
            profile_peak_util.set_text(f"{max(_profile_state['util'] or [0]):.0f}%")
            profile_peak_power.set_text(f'{pp:.0f} W' if pp else '--')
            profile_peak_rss.set_text(f'{pr:,.0f} MB' if pr else '--')

    def _load_profile_from_csv(csv_path):
        """Populate the profile charts from a completed CSV (post-run view)."""
        if not csv_path or not os.path.exists(csv_path):
            for k in ('timestamps', 'util', 'temp', 'power', 'vram_gb',
                      'kv_pct', 'tps', 'running', 'waiting'):
                _profile_state[k].clear()
            _apply_profile_charts()
            profile_csv_label.set_text('No profile CSV for this run')
            return
        data = benchmark_service.read_profile_csv(csv_path, max_points=_PROFILE_MAX_POINTS)
        _profile_state['run_name'] = None  # stop live polling
        _profile_state['csv_path'] = csv_path
        _profile_state['timestamps'].clear(); _profile_state['timestamps'].extend(data.get('timestamps', []))
        _profile_state['util'].clear(); _profile_state['util'].extend(data.get('gpu_util_pct', []))
        _profile_state['temp'].clear(); _profile_state['temp'].extend(data.get('gpu_temp_c', []))
        _profile_state['power'].clear(); _profile_state['power'].extend(data.get('gpu_power_w', []))
        _profile_state['vram_gb'].clear()
        _profile_state['vram_gb'].extend([round(x/1024.0, 2) for x in data.get('gpu_mem_mb', [])])
        _profile_state['kv_pct'].clear(); _profile_state['kv_pct'].extend(data.get('kv_cache_pct', []))
        _profile_state['tps'].clear(); _profile_state['tps'].extend(data.get('tokens_per_sec', []) or [0]*len(data.get('timestamps', [])))
        _profile_state['running'].clear(); _profile_state['running'].extend(data.get('requests_running', []))
        _profile_state['waiting'].clear(); _profile_state['waiting'].extend(data.get('requests_waiting', []))
        profile_csv_label.set_text(f'CSV: {csv_path}')
        _apply_profile_charts()

    async def refresh_servers():
        running = vllm_service.list_running()
        opts = {port: f":{info['port']} — {info['model']}" for port, info in running.items()}
        server_select.options = opts
        server_select.update()
        if len(opts) == 1:
            server_select.value = next(iter(opts.keys()))
        elif opts and not server_select.value:
            server_select.value = next(iter(opts.keys()))

    def refresh_saved_results():
        _saved_results_cache.clear()
        results = benchmark_service.list_saved_results()
        _saved_results_cache.extend(results)
        opts = {r['path']: f"{r['name']} ({r['type']}) - {r['timestamp'][:16]}" for r in results}
        prev_select.options = opts
        prev_select.update()
        cmp_a.options = opts
        cmp_a.update()
        cmp_b.options = opts
        cmp_b.update()

    def apply_preset_quick_perf():
        _highlight_preset(preset_quick_perf)
        perf_enable.value = True
        qual_enable.value = False
        dataset_select.value = 'random'
        num_prompts_input.value = 100
        request_rate_input.value = 'inf'
        max_concurrency_input.value = 64
        random_input_len.value = 512
        random_output_len.value = 256
        toggle_random_fields()

    def apply_preset_full_perf():
        _highlight_preset(preset_full_perf)
        perf_enable.value = True
        qual_enable.value = False
        dataset_select.value = 'sharegpt'
        num_prompts_input.value = 1000
        request_rate_input.value = 'inf'
        max_concurrency_input.value = 64
        toggle_random_fields()

    def apply_preset_quick_qual():
        _highlight_preset(preset_quick_qual)
        perf_enable.value = False
        qual_enable.value = True
        for t, cb in task_checks.items():
            cb.value = (t == 'mmlu')
        num_fewshot_input.value = 5
        num_concurrent_input.value = 16
        limit_input.value = 0

    def apply_preset_full_qual():
        _highlight_preset(preset_full_qual)
        perf_enable.value = False
        qual_enable.value = True
        for cb in task_checks.values():
            cb.value = True
        num_fewshot_input.value = 5
        num_concurrent_input.value = 16
        limit_input.value = 0

    def apply_preset_max_ctx():
        _highlight_preset(preset_max_ctx)
        perf_enable.value = False
        qual_enable.value = False
        ctx_enable.value = True
        ctx_upper_input.value = 131072
        ctx_step_input.value = 4096

    def _capture_script_state():
        return {
            'perf': {
                'enabled': bool(perf_enable.value),
                'dataset': dataset_select.value,
                'num_prompts': int(num_prompts_input.value or 0),
                'request_rate': str(request_rate_input.value),
                'max_concurrency': int(max_concurrency_input.value or 0),
                'random_input_len': int(random_input_len.value or 0),
                'random_output_len': int(random_output_len.value or 0),
            },
            'quality': {
                'enabled': bool(qual_enable.value),
                'tasks': [t for t, cb in task_checks.items() if cb.value],
                'num_fewshot': int(num_fewshot_input.value or 0),
                'num_concurrent': int(num_concurrent_input.value or 0),
                'limit': int(limit_input.value or 0),
            },
            'context_sweep': {
                'enabled': bool(ctx_enable.value),
                'upper_bound': int(ctx_upper_input.value or 0),
                'step': int(ctx_step_input.value or 0),
            },
        }

    def _apply_script_state(cfg):
        perf = cfg.get('perf', {})
        perf_enable.value = bool(perf.get('enabled', False))
        if 'dataset' in perf:
            dataset_select.value = perf['dataset']
        if 'num_prompts' in perf:
            num_prompts_input.value = perf['num_prompts']
        if 'request_rate' in perf:
            request_rate_input.value = perf['request_rate']
        if 'max_concurrency' in perf:
            max_concurrency_input.value = perf['max_concurrency']
        if 'random_input_len' in perf:
            random_input_len.value = perf['random_input_len']
        if 'random_output_len' in perf:
            random_output_len.value = perf['random_output_len']
        toggle_random_fields()
        qual = cfg.get('quality', {})
        qual_enable.value = bool(qual.get('enabled', False))
        wanted_tasks = set(qual.get('tasks', []))
        for t, cb in task_checks.items():
            cb.value = (t in wanted_tasks)
        if 'num_fewshot' in qual:
            num_fewshot_input.value = qual['num_fewshot']
        if 'num_concurrent' in qual:
            num_concurrent_input.value = qual['num_concurrent']
        if 'limit' in qual:
            limit_input.value = qual['limit']
        ctx = cfg.get('context_sweep', {})
        ctx_enable.value = bool(ctx.get('enabled', False))
        if 'upper_bound' in ctx:
            ctx_upper_input.value = ctx['upper_bound']
        if 'step' in ctx:
            ctx_step_input.value = ctx['step']

    def refresh_scripts():
        items = benchmark_service.list_scripts()
        opts = {it['path']: it['name'] for it in items}
        script_select.options = opts
        script_select.update()

    def on_script_save():
        name = (script_name_input.value or run_name_input.value or '').strip()
        if not name:
            ui.notify('Enter a script name', type='warning')
            return
        benchmark_service.save_script(name, _capture_script_state())
        ui.notify(f'Saved script: {name}', type='positive')
        refresh_scripts()

    def on_script_load():
        path = script_select.value
        if not path:
            ui.notify('Select a script first', type='warning')
            return
        cfg = benchmark_service.load_script(path)
        if not cfg:
            ui.notify('Script not found', type='negative')
            return
        _apply_script_state(cfg)
        script_name_input.value = cfg.get('name', '')
        ui.notify(f'Loaded script: {cfg.get("name", "?")}', type='info')

    _axis_labels = {
        'quality_score': 'Quality',
        'throughput_tps': 'Output tokens/sec',
        'prefill_tps': 'Prefill tokens/sec',
        'parameters_b': 'Parameters (B)',
        'size_gb': 'Model size (GB)',
        'peak_vram_gb': 'Peak VRAM (GB)',
        'avg_gpu_power_w': 'Avg GPU power (W)',
    }

    # Direction of "better" per axis: True = higher is better, False = lower is better.
    _axis_higher_is_better = {
        'quality_score': True,
        'throughput_tps': True,
        'prefill_tps': True,
        'parameters_b': False,
        'size_gb': False,
        'peak_vram_gb': False,
        'avg_gpu_power_w': False,
    }

    def _pareto_frontier_points(points, x_hi_better, y_hi_better):
        # points: list of [x, y, label]. Returns non-dominated subset sorted by x.
        frontier = []
        for i, p in enumerate(points):
            dominated = False
            for j, q in enumerate(points):
                if i == j:
                    continue
                x_ge = q[0] >= p[0] if x_hi_better else q[0] <= p[0]
                y_ge = q[1] >= p[1] if y_hi_better else q[1] <= p[1]
                x_gt = q[0] > p[0] if x_hi_better else q[0] < p[0]
                y_gt = q[1] > p[1] if y_hi_better else q[1] < p[1]
                if x_ge and y_ge and (x_gt or y_gt):
                    dominated = True
                    break
            if not dominated:
                frontier.append(p)
        frontier.sort(key=lambda p: p[0])
        return frontier

    def refresh_pareto():
        tasks = benchmark_service.list_quality_tasks_seen()
        pareto_task_select.options = tasks
        if tasks and pareto_task_select.value not in tasks:
            pareto_task_select.value = tasks[0]
        pareto_task_select.update()
        quants = benchmark_service.list_quantizations_seen()
        current_q = list(pareto_quant_filter.value or [])
        pareto_quant_filter.options = quants
        pareto_quant_filter.value = [q for q in current_q if q in quants]
        pareto_quant_filter.update()
        run_names = benchmark_service.list_run_names_seen()
        current_r = list(pareto_run_filter.value or [])
        pareto_run_filter.options = run_names
        pareto_run_filter.value = [r for r in current_r if r in run_names]
        pareto_run_filter.update()

        metric = pareto_task_select.value if pareto_task_select.value in tasks else None
        x_key = pareto_xaxis_select.value or 'throughput_tps'
        y_key = pareto_yaxis_select.value or 'quality_score'
        color_by = pareto_colorby_select.value or 'quantization'
        allow_quants = set(pareto_quant_filter.value or [])
        allow_runs = set(pareto_run_filter.value or [])

        rows = benchmark_service.build_pareto_dataset(metric)
        groups = {}
        skipped_empty_axis = 0
        for r in rows:
            if allow_runs and r.get('run_name') not in allow_runs:
                continue
            q = (r.get('quantization') or 'UNKNOWN').upper()
            if allow_quants and q not in allow_quants:
                continue
            x = r.get(x_key)
            y = r.get(y_key)
            if x is None or y is None:
                skipped_empty_axis += 1
                continue
            if color_by == 'model_family':
                mid = r.get('model_id') or r.get('run_name') or ''
                group_key = (mid.split('/', 1)[0] if '/' in mid else mid) or 'unknown'
            else:
                group_key = q
            groups.setdefault(group_key, []).append([x, y, r.get('run_name', '')])

        series = []
        for key, pts in sorted(groups.items()):
            series.append({
                'name': key,
                'type': 'scatter',
                'symbolSize': 14,
                'data': pts,
                'label': {'show': True, 'position': 'right', 'formatter': '{@[2]}', 'fontSize': 10},
            })

        # Pareto frontier line: drawn once there's more than one plotted point.
        all_pts = [pt for pts in groups.values() for pt in pts]
        if len(all_pts) > 1:
            frontier = _pareto_frontier_points(
                all_pts,
                _axis_higher_is_better.get(x_key, True),
                _axis_higher_is_better.get(y_key, True),
            )
            if len(frontier) >= 2:
                series.append({
                    'name': 'Pareto frontier',
                    'type': 'line',
                    'data': [[p[0], p[1]] for p in frontier],
                    'smooth': True,
                    'showSymbol': False,
                    'lineStyle': {'type': 'dashed', 'width': 2, 'color': '#888'},
                    'itemStyle': {'color': '#888'},
                    'tooltip': {'show': False},
                    'z': 1,
                })
        pareto_chart.options['series'] = series
        x_label = _axis_labels.get(x_key, x_key)
        y_label = _axis_labels.get(y_key, y_key)
        pareto_chart.options['xAxis'] = {'type': 'value', 'name': x_label}
        # Only force 0-1 range for quality axis.
        if y_key == 'quality_score':
            pareto_chart.options['yAxis'] = {'type': 'value', 'name': y_label, 'min': 0, 'max': 1}
        else:
            pareto_chart.options['yAxis'] = {'type': 'value', 'name': y_label}
        pareto_chart.options['title']['text'] = f'{y_label} vs {x_label}'
        pareto_chart.update()

        total_plotted = sum(len(pts) for pts in groups.values())
        if total_plotted == 0:
            hint = []
            if y_key == 'quality_score' and not any(r.get('has_quality') for r in rows):
                hint.append('no saved quality runs yet — try Y=Throughput')
            if x_key in ('peak_vram_gb', 'avg_gpu_power_w') and not any(r.get(x_key) for r in rows):
                hint.append('no profile data saved on any run for that axis')
            if allow_runs and not rows:
                hint.append('run filter excludes everything')
            pareto_empty_label.set_text(
                'No points to plot. ' + ('; '.join(hint) if hint else f'{skipped_empty_axis} rows missing the selected axis.')
            )
        else:
            pareto_empty_label.set_text(f'{total_plotted} point(s) across {len(groups)} group(s).')

    def on_script_delete():
        path = script_select.value
        if not path:
            ui.notify('Select a script first', type='warning')
            return
        benchmark_service.delete_script(path)
        ui.notify('Script deleted', type='info')
        refresh_scripts()

    def _set_running(is_running):
        run_btn.set_enabled(not is_running)
        stop_btn.visible = is_running
        bench_log.visible = True
        progress_bar.visible = is_running
        progress_label.visible = is_running
        if is_running:
            progress_bar.value = 0
            progress_label.set_text('')

    def _update_progress(line):
        """Parse tqdm-style progress from a line and update the progress bar."""
        # Match tqdm patterns like "67%|██..." or " 67%|"
        m = re.search(r'(\d+)%\|', line)
        if m:
            pct = int(m.group(1))
            progress_bar.value = pct / 100.0
            progress_label.set_text(f'{pct}%')

    def _fmt(val):
        if val is None:
            return '-'
        if isinstance(val, float):
            return f'{val:.4f}'
        return str(val)

    def _show_perf_result(parsed, table):
        rows = []
        rows.append({'metric': 'Request Throughput (req/s)', 'value': _fmt(parsed.get('request_throughput'))})
        rows.append({'metric': 'Output Throughput (tok/s)', 'value': _fmt(parsed.get('output_throughput'))})
        rows.append({'metric': 'Prefill Throughput (tok/s)', 'value': _fmt(parsed.get('prefill_throughput'))})
        for metric in ['ttft', 'tpot', 'itl', 'e2el']:
            mdata = parsed.get('metrics', {}).get(metric, {})
            for p in ['mean', 'p50', 'p75', 'p90', 'p95', 'p99']:
                if p in mdata:
                    rows.append({'metric': f'{metric.upper()} {p} (ms)', 'value': _fmt(mdata[p])})
        table.rows = rows
        table.update()
        table.visible = True

    def _show_ctx_result(parsed, table, headline):
        rows = []
        for p in parsed.get('probes', []):
            rows.append({
                'n': p.get('n'),
                'success': '✓' if p.get('success') else '✗',
                'ttft_ms': _fmt(p.get('ttft_ms')),
                'kv_cache_pct_at_end': _fmt(p.get('kv_cache_pct_at_end')),
                'error': (p.get('error') or '')[:80],
            })
        table.rows = rows
        table.update()
        table.visible = True
        max_ctx = parsed.get('max_context_tokens')
        headline.set_text(f'Max context: {max_ctx:,} tokens' if max_ctx else 'Max context: --')
        headline.visible = True

    def _show_qual_result(parsed, table):
        rows = []
        for t in parsed.get('tasks', []):
            acc_val = t.get('acc_norm', t.get('acc', t.get('exact_match')))
            stderr = t.get('acc_norm_stderr', t.get('acc_stderr', t.get('exact_match_stderr')))
            rows.append({
                'task': t['task'],
                'acc': _fmt(acc_val),
                'stderr': _fmt(stderr),
            })
        table.rows = rows
        table.update()
        table.visible = True

    def _read_chunk(fd):
        """Read available bytes from file descriptor."""
        try:
            return os.read(fd, 8192)
        except OSError:
            return b''

    def _client_alive():
        try:
            bench_log.client.check_existence()
            return True
        except Exception:
            return False

    async def _stream_proc(proc):
        """Stream subprocess output to log and progress bar."""
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
                    bench_log.push(stripped)
                    _update_progress(stripped)
            await asyncio.sleep(0)
        if not _client_alive():
            return
        # Drain remaining
        while True:
            raw = await run.io_bound(_read_chunk, fd)
            if not raw:
                break
            buf += raw.decode('utf-8', errors='replace')
        for part in re.split(r'[\r\n]', buf):
            stripped = part.strip()
            if stripped:
                bench_log.push(stripped)
                _update_progress(stripped)

    async def run_benchmark():
        do_perf = perf_enable.value
        do_qual = qual_enable.value
        do_ctx = ctx_enable.value
        if not do_perf and not do_qual and not do_ctx:
            ui.notify('Enable at least one benchmark type', type='warning')
            return
        port = server_select.value
        if not port:
            ui.notify('Select a server first', type='warning')
            return
        if do_qual:
            selected_tasks = [t for t, cb in task_checks.items() if cb.value]
            if not selected_tasks:
                ui.notify('Select at least one quality task', type='warning')
                return
        rname = run_name_input.value.strip() or 'unnamed'
        running = vllm_service.list_running()
        model = running.get(port, {}).get('model', '')
        if not model:
            ui.notify('Could not determine model name', type='warning')
            return

        _set_running(True)
        bench_log.clear()

        try:
            model_meta = await run.io_bound(benchmark_service.get_model_metadata, model, port)
        except Exception as e:
            bench_log.push(f'[warn] model metadata lookup failed: {e}')
            model_meta = {'model_id': model}
        try:
            csv_path = metrics_service.start_run_recording(port, rname)
        except Exception as e:
            bench_log.push(f'[warn] profile recording failed to start: {e}')
            csv_path = None
        _reset_profile_buffers(rname, csv_path)
        profile_status.set_text(f'Recording — {rname}' if csv_path else f'No profile — {rname}')
        profile_exp.open()

        def _build_extras_safely():
            try:
                summary = benchmark_service.summarize_profile_csv(csv_path) if csv_path else {}
            except Exception:
                summary = {}
            return {
                'model_meta': model_meta or {},
                'profile_csv': csv_path,
                'profile_summary': summary,
            }

        async def _run_sub(label, builder_kwargs, launcher, parser, show_fn):
            """Run one sub-benchmark in isolation so its failure can't skip the others."""
            run_status.set_text(f'Running {label}...')
            progress_bar.value = 0
            progress_label.set_text('')
            result_dir = tempfile.mkdtemp(prefix=builder_kwargs['prefix'])
            try:
                proc = await run.io_bound(launcher, result_dir)
            except Exception as e:
                ui.notify(f'{label}: could not launch ({e})', type='negative')
                bench_log.push(f'[error] {label} launch: {e}')
                return
            try:
                await _stream_proc(proc)
            except Exception as e:
                bench_log.push(f'[warn] {label} stream error: {e}')

            if proc.returncode != 0:
                ui.notify(f'{label} failed (exit code {proc.returncode})', type='negative')
                return

            try:
                parsed = await run.io_bound(parser, result_dir, rname, _build_extras_safely())
            except Exception as e:
                ui.notify(f'{label}: parse error ({e})', type='negative')
                bench_log.push(f'[error] {label} parse: {e}')
                return
            if not parsed:
                ui.notify(f'{label} finished but no result JSON found in {result_dir}',
                          type='warning')
                bench_log.push(f'[warn] {label} result dir had no JSON: {result_dir}')
                return
            if show_fn:
                try:
                    show_fn(parsed)
                except Exception as e:
                    bench_log.push(f'[warn] {label} show error: {e}')
            ui.notify(f'{label} complete', type='positive')

        try:
            if do_perf:
                def _launch_perf(result_dir):
                    return benchmark_service.run_perf_benchmark(
                        port, model, dataset_select.value,
                        int(num_prompts_input.value),
                        float(request_rate_input.value),
                        int(max_concurrency_input.value),
                        int(random_input_len.value),
                        int(random_output_len.value),
                        result_dir, rname,
                    )
                await _run_sub(
                    'Performance benchmark',
                    {'prefix': 'bench_perf_'},
                    _launch_perf,
                    benchmark_service.parse_perf_result,
                    lambda parsed: _show_perf_result(parsed, perf_table),
                )

            if do_ctx:
                def _launch_ctx(result_dir):
                    return benchmark_service.run_context_sweep(
                        port, model, result_dir, rname,
                        int(ctx_upper_input.value),
                        int(ctx_step_input.value),
                    )
                await _run_sub(
                    'Context sweep',
                    {'prefix': 'bench_ctx_'},
                    _launch_ctx,
                    benchmark_service.parse_context_sweep_result,
                    lambda parsed: _show_ctx_result(parsed, ctx_table, ctx_headline),
                )

            if do_qual:
                def _launch_qual(result_dir):
                    return benchmark_service.run_quality_benchmark(
                        port, model, selected_tasks,
                        int(num_fewshot_input.value),
                        int(num_concurrent_input.value),
                        int(limit_input.value),
                        result_dir, rname,
                    )
                await _run_sub(
                    'Quality benchmark',
                    {'prefix': 'bench_qual_'},
                    _launch_qual,
                    benchmark_service.parse_quality_result,
                    lambda parsed: _show_qual_result(parsed, qual_table),
                )

            progress_bar.value = 1.0
            progress_label.set_text('100%')
            run_status.set_text('Done')
        except Exception as e:
            if _client_alive():
                run_status.set_text(f'Error: {e}')
                ui.notify(f'Error: {e}', type='negative')
                bench_log.push(f'[error] outer run_benchmark: {e}')
        finally:
            try:
                metrics_service.stop_run_recording(rname)
            except Exception:
                pass
            if _client_alive():
                profile_status.set_text('Idle (last run complete)')
                _set_running(False)
                refresh_saved_results()
                refresh_pareto()

    def on_stop():
        benchmark_service.stop_benchmark()
        run_status.set_text('Benchmark stopped')
        ui.notify('Benchmark stopped', type='info')
        _set_running(False)

    def on_load():
        path = prev_select.value
        if not path:
            ui.notify('Select a result to load', type='warning')
            return
        data = benchmark_service.load_result(path)
        prev_perf_table.visible = False
        prev_qual_table.visible = False
        if data.get('type') == 'perf':
            _show_perf_result(data, prev_perf_table)
        elif data.get('type') == 'quality':
            _show_qual_result(data, prev_qual_table)
        elif data.get('type') == 'context_sweep':
            _show_ctx_result(data, ctx_table, ctx_headline)
        csv_path = data.get('profile_csv')
        if csv_path:
            _load_profile_from_csv(csv_path)
            profile_exp.open()
            profile_status.set_text(f"Loaded profile for {data.get('run_name', '?')}")
        else:
            profile_status.set_text(f"No profile for {data.get('run_name', '?')}")

    def on_compare():
        pa = cmp_a.value
        pb = cmp_b.value
        if not pa or not pb:
            ui.notify('Select two results to compare', type='warning')
            return
        comp = benchmark_service.compare_results(pa, pb)
        rows = []
        for r in comp.get('rows', []):
            a_val = r.get('a')
            b_val = r.get('b')
            if a_val is not None and b_val is not None:
                try:
                    delta = float(b_val) - float(a_val)
                    delta_str = f'{delta:+.4f}'
                except (ValueError, TypeError):
                    delta_str = '-'
            else:
                delta_str = '-'
            rows.append({
                'metric': r['metric'],
                'a': _fmt(a_val),
                'b': _fmt(b_val),
                'delta': delta_str,
            })
        cmp_table.columns[1]['label'] = f'Run A: {comp["run_a"]}'
        cmp_table.columns[2]['label'] = f'Run B: {comp["run_b"]}'
        cmp_table.rows = rows
        cmp_table.update()
        cmp_table.visible = True

    # Wire up callbacks
    refresh_btn.on_click(refresh_servers)
    preset_quick_perf.on_click(apply_preset_quick_perf)
    preset_full_perf.on_click(apply_preset_full_perf)
    preset_quick_qual.on_click(apply_preset_quick_qual)
    preset_full_qual.on_click(apply_preset_full_qual)
    preset_max_ctx.on_click(apply_preset_max_ctx)
    def apply_preset_custom():
        _highlight_preset(preset_custom)
        advanced.open()
    preset_custom.on_click(apply_preset_custom)
    run_btn.on_click(run_benchmark)
    stop_btn.on_click(on_stop)
    load_btn.on_click(on_load)
    refresh_results_btn.on_click(refresh_saved_results)
    cmp_btn.on_click(on_compare)

    def on_backfill_metadata():
        port = server_select.value
        updated = benchmark_service.backfill_metadata(port_for_running_model=port)
        ui.notify(f'Backfilled metadata on {updated} result file(s)', type='positive')
        refresh_pareto()

    script_save_btn.on_click(on_script_save)
    script_load_btn.on_click(on_script_load)
    script_delete_btn.on_click(on_script_delete)

    def _update_queue_count():
        q = queue_service.get_queue()
        n = len(q.get('benchmarks', [])) if q else 0
        queue_count_label.set_text(f'({n} in queue)' if n else '')

    def add_to_queue():
        if queue_service.is_running():
            ui.notify('Cannot modify queue while it is running', type='warning')
            return
        cfg = _capture_script_state()
        if not any((cfg.get(t) or {}).get('enabled') for t in ('perf', 'quality', 'context_sweep')):
            ui.notify('Enable at least one benchmark type (perf / quality / context_sweep)', type='warning')
            return
        name = (script_name_input.value or '').strip()
        if not name:
            # Auto-name: enabled types joined + timestamp
            from datetime import datetime
            enabled = [t for t in ('perf', 'quality', 'context_sweep') if (cfg.get(t) or {}).get('enabled')]
            name = '+'.join(enabled) + '_' + datetime.now().strftime('%H%M%S')
        cfg['name'] = name
        try:
            queue_service.add_benchmark(name, cfg)
            ui.notify(f'Added benchmark "{name}" to queue', type='positive')
            _update_queue_count()
        except Exception as e:
            ui.notify(f'Failed: {e}', type='negative')

    queue_add_btn.on_click(add_to_queue)
    ui.timer(5.0, _update_queue_count)
    _update_queue_count()
    pareto_refresh_btn.on_click(refresh_pareto)
    pareto_task_select.on_value_change(lambda _: refresh_pareto())
    pareto_xaxis_select.on_value_change(lambda _: refresh_pareto())
    pareto_yaxis_select.on_value_change(lambda _: refresh_pareto())
    pareto_colorby_select.on_value_change(lambda _: refresh_pareto())
    pareto_quant_filter.on_value_change(lambda _: refresh_pareto())
    pareto_run_filter.on_value_change(lambda _: refresh_pareto())
    pareto_backfill_btn.on_click(on_backfill_metadata)

    ui.timer(2.0, refresh_servers)
    ui.timer(0.1, refresh_saved_results, once=True)
    ui.timer(0.1, refresh_scripts, once=True)
    ui.timer(0.5, refresh_pareto, once=True)
    ui.timer(1.0, poll_profile)

    return refresh_servers
