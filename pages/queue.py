"""Queue tab: configure, save/load, run, and monitor serial benchmark queues."""
import os

from nicegui import ui, app, run

from services import benchmark_service, hf_service, notify_service, queue_service


def _fmt_sec(s):
    if s is None:
        return '—'
    try:
        s = max(0, int(s))
    except Exception:
        return '—'
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f'{h}h{m:02d}m{sec:02d}s'
    if m:
        return f'{m}m{sec:02d}s'
    return f'{sec}s'


def _status_color(status):
    return {
        'pending': 'grey',
        'launching': 'blue',
        'running': 'blue',
        'completed': 'positive',
        'failed': 'negative',
        'cancelled': 'warning',
    }.get(status, 'grey')


def content():
    """Returns a refresh coroutine for tab-switch triggering."""
    _model_info = {}
    _scripts = {}  # name -> path
    _last_log_len = {'n': 0}

    with ui.column().classes('w-full gap-4'):

        # ── Retry banner ──
        retry_banner = ui.row().classes('w-full bg-amber-2 p-3 rounded items-center gap-3')
        retry_banner.visible = False
        with retry_banner:
            retry_msg = ui.label('').classes('text-body2')
            ui.space()
            retry_load_btn = ui.button('Load retry queue', icon='replay').props('dense')

        # ── Section 1: Job Editor ──
        ui.label('Job Editor').classes('text-subtitle1 font-bold')
        with ui.card().classes('w-full'):
            model_select = ui.select([], label='Model(s) — multi-select', multiple=True, with_input=True).classes('w-full').props('use-chips')
            with ui.row().classes('gap-4 items-end flex-wrap'):
                cache_mode = ui.radio(['KV Cache (GB)', 'GPU Mem %'], value='KV Cache (GB)').props('inline')
                kv_cache_input = ui.number('KV Cache (GB)', value=10, min=1, max=96, step=1).classes('w-32')
                gpu_slider = ui.slider(min=0.1, max=1.0, step=0.05, value=0.90).classes('w-48')
                gpu_label = ui.label().bind_text_from(gpu_slider, 'value', backward=lambda v: f'GPU Mem: {v:.0%}')
                gpu_slider.visible = False
                gpu_label.visible = False

                def _toggle_cache():
                    is_gb = cache_mode.value == 'KV Cache (GB)'
                    kv_cache_input.visible = is_gb
                    gpu_slider.visible = not is_gb
                    gpu_label.visible = not is_gb
                cache_mode.on_value_change(lambda _: _toggle_cache())

            with ui.row().classes('gap-4 items-end flex-wrap'):
                dtype_select = ui.select(['auto', 'float16', 'bfloat16'], value='auto', label='DType').classes('w-36')
                quant_select = ui.select(['', 'awq', 'gptq', 'fp8', 'bitsandbytes'], value='', label='Quantization').classes('w-36')
                trust_remote = ui.checkbox('Trust remote code', value=False)

            scripts_select = ui.select([], label='Benchmark scripts (multi-select)', multiple=True, with_input=True).classes('w-full').props('use-chips')
            ui.label('Each selected script contributes whichever of perf / quality / context_sweep it has enabled.').classes('text-xs text-grey')

            with ui.expansion('Timeout overrides (per job)', icon='schedule').classes('w-full'):
                ui.label('Soft cap: expected finish time. After it, idle window starts. If no GPU/CPU activity for idle_window, kill.').classes('text-xs text-grey')
                ui.label('Hard cap: always kills, no exceptions.').classes('text-xs text-grey')
                d = queue_service.DEFAULT_TIMEOUTS
                with ui.row().classes('gap-4 items-end flex-wrap'):
                    to_launch_soft = ui.number('Launch soft (s)', value=d['launch_soft_s'], min=30).classes('w-36')
                    to_launch_hard = ui.number('Launch hard (s)', value=d['launch_hard_s'], min=60).classes('w-36')
                    to_launch_idle = ui.number('Launch idle window (s)', value=d['launch_idle_window_s'], min=30).classes('w-44')
                with ui.row().classes('gap-4 items-end flex-wrap'):
                    to_bench_soft = ui.number('Bench soft (s)', value=d['bench_soft_s'], min=60).classes('w-36')
                    to_bench_hard = ui.number('Bench hard (s)', value=d['bench_hard_s'], min=60).classes('w-36')
                    to_bench_idle = ui.number('Bench idle window (s)', value=d['bench_idle_window_s'], min=30).classes('w-44')

            add_btn = ui.button('Add job(s) to queue', icon='add').props('color=primary')

        # ── Section 2: Current Queue ──
        ui.label('Current Queue').classes('text-subtitle1 font-bold mt-2')
        with ui.card().classes('w-full'):
            with ui.row().classes('items-end gap-2 flex-wrap w-full'):
                queue_name_input = ui.input('Queue name', value='').classes('w-64')
                save_preset_btn = ui.button('Save as preset', icon='save').props('dense outline')
                preset_select = ui.select([], label='Load preset', with_input=True).classes('w-64')
                load_preset_btn = ui.button('Load', icon='folder_open').props('dense outline')
                delete_preset_btn = ui.button('Delete preset', icon='delete').props('dense outline color=negative')
            with ui.row().classes('items-end gap-2 flex-wrap w-full'):
                retry_select = ui.select([], label='Load retry queue', with_input=True).classes('w-80')
                load_retry_btn = ui.button('Load', icon='folder_open').props('dense outline')

            queue_table = ui.table(
                columns=[
                    {'name': 'idx', 'label': '#', 'field': 'idx', 'align': 'right'},
                    {'name': 'model', 'label': 'Model', 'field': 'model', 'align': 'left'},
                    {'name': 'scripts', 'label': 'Benchmarks', 'field': 'scripts', 'align': 'left'},
                    {'name': 'status', 'label': 'Status', 'field': 'status', 'align': 'center'},
                    {'name': 'elapsed', 'label': 'Elapsed', 'field': 'elapsed', 'align': 'right'},
                    {'name': 'error', 'label': 'Error', 'field': 'error', 'align': 'left'},
                    {'name': 'remove', 'label': '', 'field': 'remove', 'align': 'center'},
                ],
                rows=[],
                row_key='id',
            ).classes('w-full')
            queue_table.add_slot('body-cell-remove', '''
                <q-td :props="props">
                  <q-btn flat dense size="sm" icon="close" color="negative"
                         @click="$parent.$emit('remove-job', props.row)" />
                </q-td>
            ''')
            queue_table.on('remove-job', lambda e: _on_remove_job(e.args.get('id')))

            with ui.row().classes('gap-2 mt-2'):
                start_btn = ui.button('Start queue', icon='play_arrow').props('color=positive')
                cancel_btn = ui.button('Stop queue', icon='stop').props('color=negative')
                clear_btn = ui.button('Clear queue', icon='delete_sweep').props('outline')

        # ── Section 3: Runner ──
        runner_card = ui.card().classes('w-full')
        with runner_card:
            ui.label('Run Status').classes('text-subtitle1 font-bold')
            with ui.row().classes('items-center gap-4 flex-wrap w-full'):
                run_model_label = ui.label('—').classes('text-body1 font-mono')
                run_step_label = ui.label('').classes('text-body2 text-grey')
                run_progress_label = ui.label('').classes('text-body2 text-grey')
            with ui.row().classes('items-center gap-6 flex-wrap'):
                cd_elapsed = ui.label('Elapsed —').classes('text-sm font-mono')
                cd_soft = ui.label('Soft —').classes('text-sm font-mono')
                cd_hard = ui.label('Hard —').classes('text-sm font-mono')
                cd_idle = ui.label('Idle —').classes('text-sm font-mono')
            ui.label('Notification channels:').classes('text-xs text-grey mt-2')
            notify_status = ui.label('—').classes('text-xs text-grey font-mono')
            ui.separator()
            log_panel = ui.log(max_lines=300).classes('w-full').style('height: 360px')
        runner_card.visible = False

        # ── Internal queue state (mirror of queue_service's queue) ──
        def _current_queue():
            q = queue_service.get_queue()
            if not q:
                queue_service.set_queue(queue_service.new_queue(name=queue_name_input.value or 'queue'))
            return queue_service.get_queue()

        def _refresh_queue_table():
            q = queue_service.get_queue()
            rows = []
            if q:
                for i, j in enumerate(q['jobs']):
                    scripts = ', '.join(j.get('benchmarks') or []) or '(none)'
                    elapsed = '—'
                    if j.get('started_at') and j.get('finished_at'):
                        try:
                            from datetime import datetime
                            a = datetime.fromisoformat(j['started_at'])
                            b = datetime.fromisoformat(j['finished_at'])
                            elapsed = _fmt_sec((b - a).total_seconds())
                        except Exception:
                            pass
                    rows.append({
                        'id': j['id'], 'idx': i + 1, 'model': j['model'],
                        'scripts': scripts, 'status': j['status'],
                        'elapsed': elapsed, 'error': (j.get('error') or '')[:120], 'remove': '',
                    })
            queue_table.rows = rows
            queue_table.update()

        def _on_remove_job(job_id):
            if queue_service.is_running():
                ui.notify('Cannot modify queue while running', type='warning')
                return
            q = queue_service.get_queue()
            if not q:
                return
            q['jobs'] = [j for j in q['jobs'] if j['id'] != job_id]
            _refresh_queue_table()

        def _add_jobs():
            if queue_service.is_running():
                ui.notify('Cannot modify queue while running', type='warning')
                return
            models = list(model_select.value or [])
            if not models:
                ui.notify('Pick at least one model', type='warning')
                return
            script_paths = list(scripts_select.value or [])
            if not script_paths:
                ui.notify('Pick at least one benchmark script', type='warning')
                return
            # Map script paths back to names
            script_names = []
            for path in script_paths:
                name = None
                for n, p in _scripts.items():
                    if p == path:
                        name = n
                        break
                script_names.append(name or os.path.splitext(os.path.basename(path))[0])

            launch = {
                'use_kv_gb': cache_mode.value == 'KV Cache (GB)',
                'kv_cache_gb': int(kv_cache_input.value or 10),
                'gpu_mem_util': float(gpu_slider.value or 0.9),
                'dtype': dtype_select.value,
                'quantization': quant_select.value or '',
                'trust_remote_code': bool(trust_remote.value),
            }
            timeouts = {
                'launch_soft_s': int(to_launch_soft.value),
                'launch_hard_s': int(to_launch_hard.value),
                'launch_idle_window_s': int(to_launch_idle.value),
                'bench_soft_s': int(to_bench_soft.value),
                'bench_hard_s': int(to_bench_hard.value),
                'bench_idle_window_s': int(to_bench_idle.value),
            }
            q = _current_queue()
            for model_id in models:
                # For local models, use the filesystem path like servers.py does
                info = _model_info.get(model_id) or {}
                real_model = info.get('path') if info.get('source') == 'local' else model_id
                j = queue_service._new_job(
                    model=real_model,
                    launch=launch,
                    benchmarks=script_names,
                    timeouts=timeouts,
                    name=model_id,
                )
                q['jobs'].append(j)
            _refresh_queue_table()
            ui.notify(f'Added {len(models)} job(s)', type='positive')

        add_btn.on_click(_add_jobs)

        # ── Save / load presets ──
        def _refresh_presets():
            items = queue_service.list_presets()
            opts = {it['path']: f'{it["name"]} ({it["num_jobs"]} jobs)' for it in items}
            preset_select.options = opts
            preset_select.update()
            ritems = queue_service.list_retries()
            ropts = {it['path']: f'{it["name"]} ({it["num_jobs"]} jobs, from {it.get("parent") or "?"})' for it in ritems}
            retry_select.options = ropts
            retry_select.update()

        def _on_save_preset():
            name = (queue_name_input.value or '').strip()
            if not name:
                ui.notify('Give the queue a name first', type='warning')
                return
            q = queue_service.get_queue()
            if not q or not q.get('jobs'):
                ui.notify('Queue is empty', type='warning')
                return
            q['name'] = name
            queue_service.save_preset(name, q)
            ui.notify(f'Preset saved: {name}', type='positive')
            _refresh_presets()

        def _on_load_preset():
            if queue_service.is_running():
                ui.notify('Cannot load while running', type='warning')
                return
            path = preset_select.value
            if not path:
                ui.notify('Pick a preset first', type='warning')
                return
            data = queue_service.load_from_path(path)
            queue_service.set_queue(data)
            queue_name_input.value = data.get('name', '')
            _refresh_queue_table()
            ui.notify(f'Loaded: {data.get("name")}', type='info')

        def _on_delete_preset():
            path = preset_select.value
            if not path:
                ui.notify('Pick a preset first', type='warning')
                return
            queue_service.delete_preset(path)
            ui.notify('Preset deleted', type='positive')
            _refresh_presets()

        def _on_load_retry():
            if queue_service.is_running():
                ui.notify('Cannot load while running', type='warning')
                return
            path = retry_select.value
            if not path:
                ui.notify('Pick a retry queue first', type='warning')
                return
            data = queue_service.load_from_path(path)
            queue_service.set_queue(data)
            queue_name_input.value = data.get('name', '')
            retry_banner.visible = False
            _refresh_queue_table()
            ui.notify(f'Loaded retry queue: {data.get("name")}', type='info')

        save_preset_btn.on_click(_on_save_preset)
        load_preset_btn.on_click(_on_load_preset)
        delete_preset_btn.on_click(_on_delete_preset)
        load_retry_btn.on_click(_on_load_retry)
        retry_load_btn.on_click(_on_load_retry)

        def _on_clear():
            if queue_service.is_running():
                ui.notify('Cannot clear while running', type='warning')
                return
            queue_service.set_queue(queue_service.new_queue(name=queue_name_input.value or 'queue'))
            _refresh_queue_table()

        clear_btn.on_click(_on_clear)

        # ── Start / Stop ──
        def _on_start():
            q = queue_service.get_queue()
            if not q or not q.get('jobs'):
                ui.notify('Queue is empty', type='warning')
                return
            pending = [j for j in q['jobs'] if j['status'] == 'pending']
            if not pending:
                ui.notify('No pending jobs in queue', type='warning')
                return
            # Persist the user-chosen queue name if they edited it
            if queue_name_input.value and queue_name_input.value.strip():
                q['name'] = queue_name_input.value.strip()
            try:
                queue_service.start()
                ui.notify('Queue started', type='positive')
                _refresh_queue_table()
                runner_card.visible = True
                _last_log_len['n'] = 0
                log_panel.clear()
            except Exception as e:
                ui.notify(f'Start failed: {e}', type='negative')

        def _on_cancel():
            if not queue_service.is_running():
                ui.notify('Queue is not running', type='info')
                return
            queue_service.cancel()
            ui.notify('Cancel requested — will stop after current step', type='warning')

        start_btn.on_click(_on_start)
        cancel_btn.on_click(_on_cancel)

        # ── Live countdown + log tick ──
        import time

        def _tick():
            q = queue_service.get_queue()
            running = queue_service.is_running()
            runner_card.visible = bool(running or (q and q.get('status') in ('running', 'completed', 'completed_with_failures', 'failed', 'cancelled')))

            # Update countdowns if a job phase is active
            current = None
            if q and q.get('current_job_index') is not None and 0 <= q['current_job_index'] < len(q['jobs']):
                current = q['jobs'][q['current_job_index']]

            if current:
                run_model_label.set_text(current['model'])
                step = current.get('current_step') or current.get('status')
                run_step_label.set_text(f'step: {step}')
                idx = q['current_job_index'] + 1
                run_progress_label.set_text(f'job {idx}/{len(q["jobs"])}')
                ps = current.get('phase_started_at')
                soft_s = current.get('phase_soft_s')
                hard_s = current.get('phase_hard_s')
                idle_w = current.get('phase_idle_window_s')
                la = current.get('phase_last_activity_at')
                crossed = current.get('phase_soft_crossed')
                if ps is not None and soft_s is not None and hard_s is not None:
                    now = time.monotonic()
                    elapsed = now - ps
                    cd_elapsed.set_text(f'Elapsed {_fmt_sec(elapsed)}')
                    cd_soft.set_text(f'Soft {_fmt_sec(soft_s - elapsed)}')
                    cd_hard.set_text(f'Hard {_fmt_sec(hard_s - elapsed)}')
                    if crossed and la is not None and idle_w is not None:
                        idle_remaining = idle_w - (now - la)
                        cd_idle.set_text(f'Idle-kill in {_fmt_sec(idle_remaining)}')
                    else:
                        cd_idle.set_text('Idle —')
                else:
                    cd_elapsed.set_text('Elapsed —')
                    cd_soft.set_text('Soft —')
                    cd_hard.set_text('Hard —')
                    cd_idle.set_text('Idle —')
            else:
                run_model_label.set_text('—')
                run_step_label.set_text('')
                run_progress_label.set_text('')
                cd_elapsed.set_text('Elapsed —')
                cd_soft.set_text('Soft —')
                cd_hard.set_text('Hard —')
                cd_idle.set_text('Idle —')

            # Notification status
            backends = notify_service.enabled_backends()
            if backends:
                notify_status.set_text(f'enabled: {", ".join(backends)}')
            else:
                notify_status.set_text('none enabled — set up in Settings')

            # Log tail incremental push
            tail = queue_service.get_log_tail(300)
            if len(tail) > _last_log_len['n']:
                for line in tail[_last_log_len['n']:]:
                    log_panel.push(line)
                _last_log_len['n'] = len(tail)
            elif len(tail) < _last_log_len['n']:
                # log was reset (probably a new queue run)
                log_panel.clear()
                for line in tail:
                    log_panel.push(line)
                _last_log_len['n'] = len(tail)

            # Table status (so live status changes show up)
            _refresh_queue_table()

            # Retry banner
            last_retry = queue_service._state.get('last_retry_path')
            if last_retry and os.path.exists(last_retry):
                retry_msg.set_text(f'Retry queue available: {os.path.basename(last_retry)} — from the latest completed run with failures.')
                retry_banner.visible = True

        ui.timer(1.0, _tick)

        # ── Initial population ──
        async def refresh_models():
            cached = await run.io_bound(hf_service.list_cached_models)
            _model_info.clear()
            for m in cached:
                _model_info[m['id']] = m
            ids = [m['id'] for m in cached]
            model_select.options = ids
            model_select.update()

        async def refresh_scripts():
            items = await run.io_bound(benchmark_service.list_scripts)
            _scripts.clear()
            for it in items:
                _scripts[it['name']] = it['path']
            opts = {it['path']: it['name'] for it in items}
            scripts_select.options = opts
            scripts_select.update()

        async def refresh_all():
            await refresh_models()
            await refresh_scripts()
            _refresh_presets()
            _refresh_queue_table()
            q = queue_service.get_queue()
            if q:
                queue_name_input.value = q.get('name') or queue_name_input.value
            _tick()

    return refresh_all
