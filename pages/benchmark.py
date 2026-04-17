import asyncio
import re
import tempfile
import os
from nicegui import ui, run
from services import vllm_service, benchmark_service


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
            stop_btn = ui.button('Stop', icon='stop').props('color=negative')
            stop_btn.visible = False
            run_status = ui.label('').classes('text-sm')

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

        # --- Pareto frontier ---
        ui.separator()
        pareto_exp = ui.expansion('Pareto Frontier (quality vs throughput)', icon='scatter_plot').classes('w-full')
        with pareto_exp:
            with ui.row().classes('items-end gap-2 w-full'):
                pareto_task_select = ui.select([], label='Quality task (y-axis)').classes('w-48')
                pareto_refresh_btn = ui.button('Refresh', icon='refresh').props('flat dense')
            pareto_chart = ui.echart({
                'title': {'text': 'Quality vs Throughput', 'textStyle': {'fontSize': 13}},
                'tooltip': {'trigger': 'item'},
                'xAxis': {'type': 'value', 'name': 'Output tokens/sec'},
                'yAxis': {'type': 'value', 'name': 'Quality', 'min': 0, 'max': 1},
                'series': [{
                    'type': 'scatter',
                    'symbolSize': 16,
                    'data': [],
                    'label': {'show': True, 'position': 'right', 'formatter': '{@[2]}'},
                }],
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

    def refresh_pareto():
        tasks = benchmark_service.list_quality_tasks_seen()
        pareto_task_select.options = tasks
        if tasks and pareto_task_select.value not in tasks:
            pareto_task_select.value = tasks[0]
        pareto_task_select.update()
        metric = pareto_task_select.value if pareto_task_select.value in tasks else None
        rows = benchmark_service.build_pareto_dataset(metric)
        points = []
        for r in rows:
            x = r.get('throughput_tps')
            y = r.get('quality_score')
            if x is None or y is None:
                continue
            points.append([x, y, r.get('run_name', '')])
        pareto_chart.options['series'][0]['data'] = points
        pareto_chart.options['title']['text'] = f"Quality ({metric or 'first available'}) vs Throughput"
        pareto_chart.update()

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
            if do_perf:
                run_status.set_text('Running performance benchmark...')
                result_dir = tempfile.mkdtemp(prefix='bench_perf_')
                proc = await run.io_bound(
                    benchmark_service.run_perf_benchmark,
                    port, model, dataset_select.value,
                    int(num_prompts_input.value),
                    float(request_rate_input.value),
                    int(max_concurrency_input.value),
                    int(random_input_len.value),
                    int(random_output_len.value),
                    result_dir, rname,
                )
                await _stream_proc(proc)
                if proc.returncode == 0:
                    parsed = await run.io_bound(benchmark_service.parse_perf_result, result_dir, rname)
                    if parsed:
                        _show_perf_result(parsed, perf_table)
                        ui.notify('Performance benchmark complete', type='positive')
                    else:
                        ui.notify('Perf finished but no result JSON found', type='warning')
                else:
                    ui.notify(f'Perf benchmark failed (exit code {proc.returncode})', type='negative')

            if do_ctx:
                run_status.set_text('Running context-length sweep...')
                progress_bar.value = 0
                progress_label.set_text('')
                result_dir = tempfile.mkdtemp(prefix='bench_ctx_')
                proc = await run.io_bound(
                    benchmark_service.run_context_sweep,
                    port, model, result_dir, rname,
                    int(ctx_upper_input.value),
                    int(ctx_step_input.value),
                )
                await _stream_proc(proc)
                if proc.returncode == 0:
                    parsed = await run.io_bound(benchmark_service.parse_context_sweep_result, result_dir, rname)
                    if parsed:
                        _show_ctx_result(parsed, ctx_table, ctx_headline)
                        ui.notify('Context sweep complete', type='positive')
                    else:
                        ui.notify('Sweep finished but no result JSON found', type='warning')
                else:
                    ui.notify(f'Context sweep failed (exit code {proc.returncode})', type='negative')

            if do_qual:
                run_status.set_text('Running quality benchmark...')
                progress_bar.value = 0
                progress_label.set_text('')
                result_dir = tempfile.mkdtemp(prefix='bench_qual_')
                proc = await run.io_bound(
                    benchmark_service.run_quality_benchmark,
                    port, model, selected_tasks,
                    int(num_fewshot_input.value),
                    int(num_concurrent_input.value),
                    int(limit_input.value),
                    result_dir, rname,
                )
                await _stream_proc(proc)
                if proc.returncode == 0:
                    parsed = await run.io_bound(benchmark_service.parse_quality_result, result_dir, rname)
                    if parsed:
                        _show_qual_result(parsed, qual_table)
                        ui.notify('Quality benchmark complete', type='positive')
                    else:
                        ui.notify('Quality finished but no result JSON found', type='warning')
                else:
                    ui.notify(f'Quality benchmark failed (exit code {proc.returncode})', type='negative')

            progress_bar.value = 1.0
            progress_label.set_text('100%')
            run_status.set_text('Done')
        except Exception as e:
            if _client_alive():
                run_status.set_text(f'Error: {e}')
                ui.notify(f'Error: {e}', type='negative')
        finally:
            if _client_alive():
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

    script_save_btn.on_click(on_script_save)
    script_load_btn.on_click(on_script_load)
    script_delete_btn.on_click(on_script_delete)
    pareto_refresh_btn.on_click(refresh_pareto)
    pareto_task_select.on_value_change(lambda _: refresh_pareto())

    ui.timer(2.0, refresh_servers)
    ui.timer(0.1, refresh_saved_results, once=True)
    ui.timer(0.1, refresh_scripts, once=True)
    ui.timer(0.5, refresh_pareto, once=True)

    return refresh_servers
