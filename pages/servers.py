import asyncio
import subprocess
import os
import signal
from nicegui import ui, app, run
from services import hf_service, vllm_service, metrics_service
import config

_webui_proc = None
WEBUI_ENV = "/opt/hpml_project/openwebui_env"
WEBUI_PORT = 3000


def content():
    """Servers panel. Returns a refresh callable for tab-switch triggering."""
    with ui.column().classes('w-full gap-4'):
        ui.label('Running Servers').classes('text-subtitle1 font-bold')
        hostname = config.APP_HOSTNAME
        server_grid = ui.table(
            columns=[
                {'name': 'endpoint', 'label': 'Endpoint', 'field': 'endpoint', 'align': 'left'},
                {'name': 'model', 'label': 'Model', 'field': 'model', 'align': 'left'},
                {'name': 'gpu_mem_util', 'label': 'GPU Mem %', 'field': 'gpu_mem_util'},
                {'name': 'dtype', 'label': 'DType', 'field': 'dtype'},
                {'name': 'quantization', 'label': 'Quantization', 'field': 'quantization'},
                {'name': 'health', 'label': 'Health', 'field': 'health'},
            ],
            rows=[],
            row_key='port',
            selection='single',
        ).classes('w-full')

        server_grid.add_slot('body-cell-endpoint', '''
            <q-td :props="props">
                <a :href="'http://' + props.value + '/docs'" target="_blank"
                   class="text-primary" style="text-decoration: none;">{{ props.value }}</a>
            </q-td>
        ''')

        with ui.row().classes('gap-2'):
            refresh_btn = ui.button('Refresh', icon='refresh')
            stop_btn = ui.button('Stop Selected', icon='stop').props('color=negative')

        ui.separator()

        ui.label('Launch New Server').classes('text-subtitle1 font-bold')
        with ui.card().classes('w-full'):
            model_select = ui.select([], label='Model (downloaded)', with_input=True).classes('w-96')
            with ui.row().classes('gap-4 items-end'):
                port_input = ui.number('Port', value=8001, min=1024, max=65535, step=1).classes('w-32')
            with ui.row().classes('gap-4 items-center'):
                cache_mode = ui.radio(['KV Cache (GB)', 'GPU Mem %'], value='KV Cache (GB)').props('inline')
                kv_cache_input = ui.number('KV Cache (GB)', value=10, min=1, max=96, step=1).classes('w-32')
                gpu_slider = ui.slider(min=0.1, max=1.0, step=0.05, value=0.90).classes('w-48')
                gpu_label = ui.label().bind_text_from(gpu_slider, 'value', backward=lambda v: f'GPU Mem: {v:.0%}')
                gpu_slider.visible = False
                gpu_label.visible = False

                def _toggle_cache_mode():
                    is_gb = cache_mode.value == 'KV Cache (GB)'
                    kv_cache_input.visible = is_gb
                    gpu_slider.visible = not is_gb
                    gpu_label.visible = not is_gb

                cache_mode.on_value_change(lambda _: _toggle_cache_mode())
                _toggle_cache_mode()
            with ui.row().classes('gap-4 items-end'):
                dtype_select = ui.select(['auto', 'float16', 'bfloat16'], value='auto', label='DType').classes('w-36')
                quant_select = ui.select(
                    ['', 'awq', 'gptq', 'fp8', 'bitsandbytes'],
                    value='',
                    label='Quantization',
                ).classes('w-36')
            with ui.row().classes('gap-4'):
                trust_remote_check = ui.checkbox('Trust remote code')
                record_check = ui.checkbox('Record metrics to CSV')
            with ui.row().classes('gap-2 items-center'):
                launch_btn = ui.button('Launch Server', icon='play_arrow').props('color=positive')
                webui_launch_btn = ui.button('Launch Open WebUI', icon='open_in_new').props('color=accent')
                webui_stop_btn = ui.button('Stop WebUI', icon='stop').props('color=negative')
                webui_stop_btn.visible = False
                webui_status = ui.label('').classes('text-sm')
            launch_status = ui.label('').classes('text-sm')
            with ui.row().classes('gap-2 items-center'):
                copy_log_btn = ui.button('Copy log', icon='content_copy').props('flat dense')
                copy_log_btn.visible = False
            launch_log = ui.log(max_lines=200).classes('w-full').style('height: 800px')
            launch_log.visible = False

            def copy_log():
                log_id = f'c{launch_log.id}'
                ui.run_javascript(f'navigator.clipboard.writeText(document.getElementById("{log_id}")?.innerText || "")')
                ui.notify('Log copied to clipboard', type='positive')

            copy_log_btn.on_click(copy_log)

        async def refresh_servers():
            running = vllm_service.list_running()
            rows = []
            for port, info in running.items():
                healthy = await run.io_bound(vllm_service.check_health, port)
                rows.append({
                    **info,
                    'endpoint': f'{hostname}:{port}',
                    'health': 'Healthy' if healthy else 'Starting...',
                })
            server_grid.rows = rows
            server_grid.update()

        async def refresh_models():
            cached = await run.io_bound(hf_service.list_cached_models)
            model_ids = [m['id'] for m in cached]
            model_select.options = model_ids
            if len(model_ids) == 1:
                model_select.value = model_ids[0]
            model_select.update()

        async def refresh_all():
            await refresh_models()
            await refresh_servers()

        async def stop_selected():
            if not server_grid.selected:
                ui.notify('Select a server first', type='warning')
                return
            row = server_grid.selected[0]
            metrics_service.stop_recording(row['port'])
            await run.io_bound(vllm_service.stop_server, row['port'])
            ui.notify(f"Stopped server on port {row['port']}", type='positive')
            await refresh_servers()

        async def launch():
            model = model_select.value
            if not model:
                ui.notify('Select a model first', type='warning')
                return
            launch_btn.disable()
            launch_status.set_text('Starting vLLM server...')
            launch_log.clear()
            launch_log.visible = False
            copy_log_btn.visible = False
            token = app.storage.general.get('hf_token', '')
            try:
                extra = ['--trust-remote-code'] if trust_remote_check.value else None
                use_kv_gb = cache_mode.value == 'KV Cache (GB)'
                port = vllm_service.launch_server(
                    model=model,
                    port=int(port_input.value),
                    gpu_mem_util=gpu_slider.value,
                    dtype=dtype_select.value,
                    quantization=quant_select.value or None,
                    extra_args=extra,
                    token=token or None,
                    kv_cache_gb=int(kv_cache_input.value) if use_kv_gb else None,
                )
                info = vllm_service.get_server_info(port)
                log_path = info.get('log_path') if info else f'/tmp/vllm_{port}.log'
                launch_status.set_text(f'Server launching on port {port}...')
                launch_log.visible = True
                copy_log_btn.visible = True
                launch_log.clear()
                last_log_len = 0
                for _ in range(100):
                    await asyncio.sleep(3)
                    log_lines = await run.io_bound(vllm_service.get_log_by_path, log_path, 500)
                    if len(log_lines) > last_log_len:
                        for line in log_lines[last_log_len:]:
                            launch_log.push(line)
                        last_log_len = len(log_lines)
                    alive = await run.io_bound(vllm_service.is_alive, port)
                    if not alive:
                        log_lines = await run.io_bound(vllm_service.get_log_by_path, log_path, 500)
                        if len(log_lines) > last_log_len:
                            for line in log_lines[last_log_len:]:
                                launch_log.push(line)
                        launch_status.set_text(f'Server on port {port} crashed')
                        ui.notify('Server failed to start. See log below.', type='negative')
                        break
                    status = await run.io_bound(vllm_service.get_latest_status, port)
                    launch_status.set_text(f'Port {port}: {status}')
                    healthy = await run.io_bound(vllm_service.check_health, port)
                    if healthy:
                        launch_status.set_text(f'Server ready at {hostname}:{port}')
                        ui.notify(f'Server ready at {hostname}:{port}', type='positive')
                        if record_check.value:
                            metrics_service.start_recording(port, model)
                            ui.notify('Metrics recording started', type='info')
                        break
                else:
                    launch_status.set_text(f'Server on port {port} may still be loading...')
                    ui.notify('Server is taking long to start.', type='warning')
                await refresh_servers()
            except Exception as e:
                launch_status.set_text(f'Error: {e}')
                ui.notify(f'Failed: {e}', type='negative')
            finally:
                launch_btn.enable()

        refresh_btn.on_click(refresh_all)
        stop_btn.on_click(stop_selected)
        launch_btn.on_click(launch)

        async def launch_webui():
            global _webui_proc
            if _webui_proc and _webui_proc.poll() is None:
                ui.notify('Open WebUI is already running', type='info')
                ui.navigate.to(f'http://{config.APP_HOSTNAME}:{WEBUI_PORT}', new_tab=True)
                return
            env_python = os.path.join(WEBUI_ENV, 'bin', 'python')
            if not os.path.exists(env_python):
                ui.notify(f'Open WebUI env not found at {WEBUI_ENV}. Install it first.', type='negative')
                return
            webui_launch_btn.props('disable')
            webui_status.set_text('Starting Open WebUI...')
            webui_bin = os.path.join(WEBUI_ENV, 'bin', 'open-webui')
            _webui_proc = subprocess.Popen(
                [webui_bin, 'serve', '--port', str(WEBUI_PORT)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env={**os.environ, 'OPENAI_API_BASE_URLS': 'http://localhost:8001/v1', 'WEBUI_SHOW_UPDATE_NOTIFICATION': 'false'},
            )
            for _ in range(30):
                await asyncio.sleep(2)
                if _webui_proc.poll() is not None:
                    webui_status.set_text('Open WebUI failed to start')
                    ui.notify('Open WebUI crashed', type='negative')
                    webui_launch_btn.props(remove='disable')
                    return
                try:
                    import requests
                    r = requests.get(f'http://localhost:{WEBUI_PORT}', timeout=2)
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
            url = f'http://{config.APP_HOSTNAME}:{WEBUI_PORT}'
            webui_status.set_text(f'Running at {url}')
            webui_stop_btn.visible = True
            webui_launch_btn.props(remove='disable')
            ui.navigate.to(url, new_tab=True)

        def stop_webui():
            global _webui_proc
            if _webui_proc and _webui_proc.poll() is None:
                _webui_proc.send_signal(signal.SIGTERM)
                try:
                    _webui_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _webui_proc.kill()
            _webui_proc = None
            webui_stop_btn.visible = False
            webui_status.set_text('')
            ui.notify('Open WebUI stopped', type='info')

        webui_launch_btn.on_click(launch_webui)
        webui_stop_btn.on_click(stop_webui)

        ui.timer(10.0, refresh_servers)
        ui.timer(0.1, refresh_models, once=True)

    return refresh_all
