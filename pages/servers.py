import asyncio
from nicegui import ui, app, run
from services import hf_service, vllm_service


def content():
    """Servers panel. Returns a refresh callable for tab-switch triggering."""
    with ui.column().classes('w-full gap-4'):
        ui.label('Running Servers').classes('text-subtitle1 font-bold')
        server_grid = ui.table(
            columns=[
                {'name': 'port', 'label': 'Port', 'field': 'port', 'sortable': True},
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

        with ui.row().classes('gap-2'):
            refresh_btn = ui.button('Refresh', icon='refresh')
            stop_btn = ui.button('Stop Selected', icon='stop').props('color=negative')

        ui.separator()

        ui.label('Launch New Server').classes('text-subtitle1 font-bold')
        with ui.card().classes('w-full'):
            model_select = ui.select([], label='Model (downloaded)', with_input=True).classes('w-96')
            with ui.row().classes('gap-4 items-end'):
                port_input = ui.number('Port', value=8001, min=1024, max=65535, step=1).classes('w-32')
                gpu_slider = ui.slider(min=0.1, max=1.0, step=0.05, value=0.90).classes('w-48')
                ui.label().bind_text_from(gpu_slider, 'value', backward=lambda v: f'GPU Mem: {v:.0%}')
            with ui.row().classes('gap-4 items-end'):
                dtype_select = ui.select(['auto', 'float16', 'bfloat16'], value='auto', label='DType').classes('w-36')
                quant_select = ui.select(
                    ['', 'awq', 'gptq', 'fp8', 'bitsandbytes'],
                    value='',
                    label='Quantization',
                ).classes('w-36')
            launch_btn = ui.button('Launch Server', icon='play_arrow').props('color=positive')
            launch_status = ui.label('').classes('text-sm')
            launch_log = ui.log(max_lines=200).classes('w-full').style('height: 800px')
            launch_log.visible = False

        async def refresh_servers():
            running = vllm_service.list_running()
            rows = []
            for port, info in running.items():
                healthy = await run.io_bound(vllm_service.check_health, port)
                rows.append({
                    **info,
                    'health': 'Healthy' if healthy else 'Starting...',
                })
            server_grid.rows = rows
            server_grid.update()

        async def refresh_models():
            cached = await run.io_bound(hf_service.list_cached_models)
            model_ids = [m['id'] for m in cached]
            model_select.options = model_ids
            model_select.update()

        async def refresh_all():
            await refresh_models()
            await refresh_servers()

        async def stop_selected():
            if not server_grid.selected:
                ui.notify('Select a server first', type='warning')
                return
            row = server_grid.selected[0]
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
            token = app.storage.general.get('hf_token', '')
            try:
                port = vllm_service.launch_server(
                    model=model,
                    port=int(port_input.value),
                    gpu_mem_util=gpu_slider.value,
                    dtype=dtype_select.value,
                    quantization=quant_select.value or None,
                    token=token or None,
                )
                log_path = f'/tmp/vllm_{port}.log'
                launch_status.set_text(f'Server launching on port {port}...')
                launch_log.visible = True
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
                        launch_status.set_text(f'Server ready on port {port}')
                        ui.notify(f'Server ready on port {port}', type='positive')
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

        ui.timer(10.0, refresh_servers)
        # Auto-load model list on first render
        ui.timer(0.1, refresh_models, once=True)

    return refresh_all
