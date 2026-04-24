"""Queue tab: view accumulated models × benchmarks, save/load presets, run + monitor.

Models and benchmarks are populated via 'Add to Queue' buttons on the Servers
and Benchmark tabs. The Queue tab itself is just a viewer + runner.
"""
import os
import time

from nicegui import ui

from services import notify_service, queue_service


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


def _launch_summary(launch):
    bits = []
    if launch.get('use_kv_gb'):
        bits.append(f'KV={launch.get("kv_cache_gb")}GB')
    else:
        bits.append(f'mem={launch.get("gpu_mem_util"):.0%}')
    if launch.get('dtype') and launch['dtype'] != 'auto':
        bits.append(launch['dtype'])
    if launch.get('quantization'):
        bits.append(launch['quantization'])
    if launch.get('cpu_offload_gb'):
        bits.append(f'offload={launch["cpu_offload_gb"]}GB')
    if launch.get('trust_remote_code'):
        bits.append('trust')
    return ', '.join(bits) or '—'


def _bench_summary(config):
    bits = []
    for t in ('perf', 'quality', 'context_sweep'):
        sub = config.get(t) or {}
        if sub.get('enabled'):
            bits.append(t)
    return '+'.join(bits) or '—'


def content():
    """Returns an async refresh callable for tab-switch triggering."""
    _last_log_len = {'n': 0}

    with ui.column().classes('w-full gap-4'):

        # ── Name + preset controls ──
        with ui.card().classes('w-full'):
            with ui.row().classes('items-end gap-2 flex-wrap w-full'):
                queue_name_input = ui.input('Queue name', value='').classes('w-64')
                rename_btn = ui.button('Rename', icon='edit').props('dense outline')
                save_preset_btn = ui.button('Save as preset', icon='save').props('dense outline')
                preset_select = ui.select([], label='Load preset', with_input=True).classes('w-96')
                load_preset_btn = ui.button('Load', icon='folder_open').props('dense outline')
                delete_preset_btn = ui.button('Delete preset', icon='delete').props('dense outline color=negative')
            ui.label('Failed jobs from a completed queue are saved as a new preset named "{queue}__failures_<timestamp>".').classes('text-xs text-grey')

        # ── Models table ──
        ui.label('Models in queue').classes('text-subtitle1 font-bold')
        ui.label('Add models from the Servers tab using the "Add to Queue" button.').classes('text-xs text-grey')
        models_table = ui.table(
            columns=[
                {'name': 'name', 'label': 'Name', 'field': 'name', 'align': 'left'},
                {'name': 'model', 'label': 'Model / Path', 'field': 'model', 'align': 'left'},
                {'name': 'launch', 'label': 'Launch', 'field': 'launch', 'align': 'left'},
                {'name': 'remove', 'label': '', 'field': 'remove', 'align': 'center'},
            ],
            rows=[], row_key='id',
        ).classes('w-full')
        models_table.add_slot('body-cell-remove', '''
            <q-td :props="props">
              <q-btn flat dense size="sm" icon="close" color="negative"
                     @click="$parent.$emit('remove-model', props.row)" />
            </q-td>
        ''')
        models_table.on('remove-model', lambda e: _on_remove_model(e.args.get('id')))
        with ui.row().classes('gap-2'):
            clear_models_btn = ui.button('Clear all models', icon='delete_sweep').props('dense outline')

        # ── Benchmarks table ──
        ui.label('Benchmarks in queue').classes('text-subtitle1 font-bold mt-2')
        ui.label('Add benchmarks from the Benchmark tab using the "Add to Queue" button.').classes('text-xs text-grey')
        benches_table = ui.table(
            columns=[
                {'name': 'name', 'label': 'Name', 'field': 'name', 'align': 'left'},
                {'name': 'types', 'label': 'Types', 'field': 'types', 'align': 'left'},
                {'name': 'remove', 'label': '', 'field': 'remove', 'align': 'center'},
            ],
            rows=[], row_key='id',
        ).classes('w-full')
        benches_table.add_slot('body-cell-remove', '''
            <q-td :props="props">
              <q-btn flat dense size="sm" icon="close" color="negative"
                     @click="$parent.$emit('remove-bench', props.row)" />
            </q-td>
        ''')
        benches_table.on('remove-bench', lambda e: _on_remove_bench(e.args.get('id')))
        with ui.row().classes('gap-2'):
            clear_benches_btn = ui.button('Clear all benchmarks', icon='delete_sweep').props('dense outline')

        # ── Expansion preview + start controls ──
        with ui.card().classes('w-full'):
            expansion_label = ui.label('').classes('text-body1 font-bold')
            with ui.row().classes('gap-2'):
                start_btn = ui.button('Start queue (all-to-all)', icon='play_arrow').props('color=positive')
                cancel_btn = ui.button('Stop queue', icon='stop').props('color=negative')

        # ── Runner / live state ──
        runner_card = ui.card().classes('w-full')
        with runner_card:
            ui.label('Run status').classes('text-subtitle1 font-bold')
            with ui.row().classes('items-center gap-4 flex-wrap w-full'):
                run_job_label = ui.label('—').classes('text-body1 font-mono')
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
            ui.label('Jobs').classes('text-xs text-grey')
            jobs_table = ui.table(
                columns=[
                    {'name': 'idx', 'label': '#', 'field': 'idx', 'align': 'right'},
                    {'name': 'name', 'label': 'Job', 'field': 'name', 'align': 'left'},
                    {'name': 'status', 'label': 'Status', 'field': 'status', 'align': 'center'},
                    {'name': 'elapsed', 'label': 'Elapsed', 'field': 'elapsed', 'align': 'right'},
                    {'name': 'error', 'label': 'Error', 'field': 'error', 'align': 'left'},
                ],
                rows=[], row_key='id',
            ).classes('w-full')
            ui.separator()
            log_panel = ui.log(max_lines=300).classes('w-full').style('height: 320px')
        runner_card.visible = False

        # ── Handlers ──
        def _current_queue():
            q = queue_service.get_queue()
            if q is None:
                queue_service.set_queue(queue_service.new_queue())
                q = queue_service.get_queue()
            return q

        def _on_remove_model(entry_id):
            try:
                queue_service.remove_model(entry_id)
            except Exception as e:
                ui.notify(str(e), type='warning'); return
            _refresh_tables()

        def _on_remove_bench(entry_id):
            try:
                queue_service.remove_benchmark(entry_id)
            except Exception as e:
                ui.notify(str(e), type='warning'); return
            _refresh_tables()

        def _on_clear_models():
            try:
                queue_service.clear_models()
            except Exception as e:
                ui.notify(str(e), type='warning'); return
            _refresh_tables()

        def _on_clear_benches():
            try:
                queue_service.clear_benchmarks()
            except Exception as e:
                ui.notify(str(e), type='warning'); return
            _refresh_tables()

        clear_models_btn.on_click(_on_clear_models)
        clear_benches_btn.on_click(_on_clear_benches)

        def _on_rename():
            new_name = (queue_name_input.value or '').strip()
            if not new_name:
                ui.notify('Enter a queue name first', type='warning'); return
            queue_service.rename_queue(new_name)
            ui.notify(f'Queue renamed to "{new_name}"', type='positive')

        rename_btn.on_click(_on_rename)

        def _refresh_presets():
            items = queue_service.list_presets()
            opts = {}
            for it in items:
                if it.get('parent'):
                    # Failure preset: highlight lineage + pre-expanded job count
                    opts[it['path']] = f'⚠ {it["name"]} ({it["num_jobs"]} failed jobs from "{it["parent"]}")'
                else:
                    opts[it['path']] = f'{it["name"]} ({it["num_models"]}m × {it["num_benchmarks"]}b)'
            preset_select.options = opts
            preset_select.update()

        def _on_save_preset():
            name = (queue_name_input.value or '').strip()
            if not name:
                ui.notify('Give the queue a name first', type='warning'); return
            q = queue_service.get_queue()
            if not q or (not q.get('models') and not q.get('jobs')):
                ui.notify('Queue is empty — add models + benchmarks first', type='warning'); return
            q['name'] = name
            queue_service.save_preset(name, q)
            ui.notify(f'Preset saved: {name}', type='positive')
            _refresh_presets()

        def _on_load_preset():
            if queue_service.is_running():
                ui.notify('Cannot load while running', type='warning'); return
            path = preset_select.value
            if not path:
                ui.notify('Pick a preset first', type='warning'); return
            data = queue_service.load_from_path(path)
            queue_service.set_queue(data)
            queue_name_input.value = data.get('name', '')
            _refresh_tables()
            ui.notify(f'Loaded preset: {data.get("name")}', type='info')

        def _on_delete_preset():
            path = preset_select.value
            if not path:
                ui.notify('Pick a preset first', type='warning'); return
            queue_service.delete_preset(path)
            ui.notify('Preset deleted', type='positive')
            _refresh_presets()

        save_preset_btn.on_click(_on_save_preset)
        load_preset_btn.on_click(_on_load_preset)
        delete_preset_btn.on_click(_on_delete_preset)

        def _on_start():
            q = queue_service.get_queue()
            if not q:
                ui.notify('No queue', type='warning'); return
            n = len(q.get('models', [])) * len(q.get('benchmarks', []))
            pending_jobs = sum(1 for j in q.get('jobs', []) if j['status'] == 'pending')
            if n == 0 and pending_jobs == 0:
                ui.notify('Need at least 1 model and 1 benchmark (or a failure preset with pending jobs)', type='warning')
                return
            if queue_name_input.value and queue_name_input.value.strip():
                queue_service.rename_queue(queue_name_input.value.strip())
            try:
                queue_service.start()
                ui.notify('Queue started', type='positive')
                runner_card.visible = True
                _last_log_len['n'] = 0
                log_panel.clear()
                _refresh_tables()
            except Exception as e:
                ui.notify(f'Start failed: {e}', type='negative')

        def _on_cancel():
            if not queue_service.is_running():
                ui.notify('Queue is not running', type='info'); return
            queue_service.cancel()
            ui.notify('Cancel requested — stops after current step', type='warning')

        start_btn.on_click(_on_start)
        cancel_btn.on_click(_on_cancel)

        # ── Refresh ──
        def _refresh_tables():
            q = queue_service.get_queue()
            # Models
            m_rows = []
            for m in (q.get('models') if q else []) or []:
                m_rows.append({
                    'id': m['id'], 'name': m['name'], 'model': m['model'],
                    'launch': _launch_summary(m['launch']), 'remove': '',
                })
            models_table.rows = m_rows
            models_table.update()
            # Benchmarks
            b_rows = []
            for b in (q.get('benchmarks') if q else []) or []:
                b_rows.append({
                    'id': b['id'], 'name': b['name'],
                    'types': _bench_summary(b['config']), 'remove': '',
                })
            benches_table.rows = b_rows
            benches_table.update()
            # Expansion preview
            if q:
                nm = len(q.get('models', []))
                nb = len(q.get('benchmarks', []))
                pending = sum(1 for j in q.get('jobs', []) if j['status'] == 'pending')
                if nm and nb:
                    expansion_label.set_text(f'Will run {nm} × {nb} = {nm * nb} jobs (all-to-all)')
                elif pending:
                    expansion_label.set_text(f'Loaded failure preset: {pending} pending job(s)')
                else:
                    expansion_label.set_text('Empty — add at least one model and one benchmark')
            else:
                expansion_label.set_text('Empty — add at least one model and one benchmark')

        def _refresh_jobs_table():
            q = queue_service.get_queue()
            rows = []
            if q:
                from datetime import datetime
                for i, j in enumerate(q.get('jobs', []) or []):
                    elapsed = '—'
                    if j.get('started_at') and j.get('finished_at'):
                        try:
                            a = datetime.fromisoformat(j['started_at'])
                            b = datetime.fromisoformat(j['finished_at'])
                            elapsed = _fmt_sec((b - a).total_seconds())
                        except Exception:
                            pass
                    rows.append({
                        'id': j['id'], 'idx': i + 1, 'name': j['name'],
                        'status': j['status'], 'elapsed': elapsed,
                        'error': (j.get('error') or '')[:120],
                    })
            jobs_table.rows = rows
            jobs_table.update()

        def _tick():
            q = queue_service.get_queue()
            running = queue_service.is_running()
            runner_card.visible = bool(running or (q and q.get('status') in ('running', 'completed', 'completed_with_failures', 'failed', 'cancelled')))

            current = None
            if q and q.get('current_job_index') is not None and 0 <= q['current_job_index'] < len(q.get('jobs', [])):
                current = q['jobs'][q['current_job_index']]

            if current:
                run_job_label.set_text(current['name'])
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
                        cd_idle.set_text(f'Idle-kill in {_fmt_sec(idle_w - (now - la))}')
                    else:
                        cd_idle.set_text('Idle —')
                else:
                    cd_elapsed.set_text('Elapsed —')
                    cd_soft.set_text('Soft —')
                    cd_hard.set_text('Hard —')
                    cd_idle.set_text('Idle —')
            else:
                run_job_label.set_text('—')
                run_step_label.set_text('')
                run_progress_label.set_text('')
                cd_elapsed.set_text('Elapsed —')
                cd_soft.set_text('Soft —')
                cd_hard.set_text('Hard —')
                cd_idle.set_text('Idle —')

            backends = notify_service.enabled_backends()
            notify_status.set_text(f'enabled: {", ".join(backends)}' if backends else 'none enabled — set up in Settings')

            tail = queue_service.get_log_tail(300)
            if len(tail) > _last_log_len['n']:
                for line in tail[_last_log_len['n']:]:
                    log_panel.push(line)
                _last_log_len['n'] = len(tail)
            elif len(tail) < _last_log_len['n']:
                log_panel.clear()
                for line in tail:
                    log_panel.push(line)
                _last_log_len['n'] = len(tail)

            _refresh_jobs_table()
            _refresh_tables()

        ui.timer(1.0, _tick)
        ui.timer(5.0, _refresh_presets)  # keep dropdowns in sync with disk

        async def refresh_all():
            _refresh_presets()
            _refresh_tables()
            q = queue_service.get_queue()
            if q and not queue_name_input.value:
                queue_name_input.value = q.get('name') or ''
            _tick()

    return refresh_all
