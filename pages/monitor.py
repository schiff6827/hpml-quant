import time
from collections import deque
from nicegui import ui, run
from services import metrics_service, vllm_service

MAX_POINTS = 150


def content():
    """Monitor tab. Returns a refresh callable."""
    with ui.column().classes('w-full gap-4'):
        with ui.row().classes('items-center gap-4'):
            ui.label('Server Monitor').classes('text-subtitle1 font-bold')
            server_select = ui.select([], label='Server', with_input=True).classes('w-64')
            refresh_btn = ui.button('Refresh servers', icon='refresh')

        # GPU hardware gauges
        ui.label('GPU').classes('text-subtitle2 font-bold')
        with ui.row().classes('gap-4 w-full'):
            with ui.card().classes('p-3'):
                ui.label('Memory').classes('text-caption')
                gpu_mem_label = ui.label('-- / -- GB').classes('text-h6')
                gpu_mem_bar = ui.linear_progress(value=0).props('instant-feedback').classes('w-48')
            with ui.card().classes('p-3'):
                ui.label('Utilization').classes('text-caption')
                gpu_util_label = ui.label('--%').classes('text-h6')
                gpu_util_bar = ui.linear_progress(value=0).props('instant-feedback').classes('w-32')
            with ui.card().classes('p-3'):
                ui.label('Temperature').classes('text-caption')
                gpu_temp_label = ui.label('--C').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('Power').classes('text-caption')
                gpu_power_label = ui.label('--W').classes('text-h6')

        # CPU memory gauges
        ui.label('CPU').classes('text-subtitle2 font-bold')
        with ui.row().classes('gap-4 w-full'):
            with ui.card().classes('p-3'):
                ui.label('RSS (current)').classes('text-caption')
                cpu_rss_label = ui.label('-- MB').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('RSS (peak)').classes('text-caption')
                cpu_rss_peak_label = ui.label('-- MB').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('Host Memory').classes('text-caption')
                host_mem_label = ui.label('-- / -- GB').classes('text-h6')
                host_mem_bar = ui.linear_progress(value=0).props('instant-feedback').classes('w-48')

        # Charts
        ui.label('Inference Metrics').classes('text-subtitle2 font-bold')
        with ui.row().classes('w-full gap-4'):
            cache_chart = ui.echart({
                'title': {'text': 'Cache Usage', 'textStyle': {'fontSize': 13}},
                'tooltip': {'trigger': 'axis'},
                'legend': {'data': ['KV Cache %', 'Prefix Hit Rate %'], 'bottom': 0},
                'xAxis': {'type': 'category', 'data': []},
                'yAxis': {'type': 'value', 'min': 0, 'max': 100, 'name': '%'},
                'series': [
                    {'name': 'KV Cache %', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False},
                    {'name': 'Prefix Hit Rate %', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False},
                ],
                'animation': False,
            }).classes('w-1/2 h-64')

            throughput_chart = ui.echart({
                'title': {'text': 'Throughput & Requests', 'textStyle': {'fontSize': 13}},
                'tooltip': {'trigger': 'axis'},
                'legend': {'data': ['Tokens/sec', 'Running', 'Waiting'], 'bottom': 0},
                'xAxis': {'type': 'category', 'data': []},
                'yAxis': [
                    {'type': 'value', 'name': 'tok/s', 'position': 'left'},
                    {'type': 'value', 'name': 'reqs', 'position': 'right'},
                ],
                'series': [
                    {'name': 'Tokens/sec', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 0},
                    {'name': 'Running', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 1},
                    {'name': 'Waiting', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 1},
                ],
                'animation': False,
            }).classes('w-1/2 h-64')

        with ui.row().classes('w-full gap-4'):
            gpu_chart = ui.echart({
                'title': {'text': 'GPU Utilization / Temp / Power', 'textStyle': {'fontSize': 13}},
                'tooltip': {'trigger': 'axis'},
                'legend': {'data': ['Utilization %', 'Temp C', 'Power W'], 'bottom': 0},
                'xAxis': {'type': 'category', 'data': []},
                'yAxis': [
                    {'type': 'value', 'min': 0, 'max': 100, 'name': '% / C', 'position': 'left'},
                    {'type': 'value', 'name': 'W', 'position': 'right'},
                ],
                'series': [
                    {'name': 'Utilization %', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 0},
                    {'name': 'Temp C', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 0},
                    {'name': 'Power W', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'yAxisIndex': 1},
                ],
                'animation': False,
            }).classes('w-1/2 h-64')

            mem_chart = ui.echart({
                'title': {'text': 'Memory: Weights vs KV', 'textStyle': {'fontSize': 13}},
                'tooltip': {'trigger': 'axis'},
                'legend': {'data': ['Weights GiB', 'KV Used GiB'], 'bottom': 0},
                'xAxis': {'type': 'category', 'data': []},
                'yAxis': {'type': 'value', 'name': 'GiB'},
                'series': [
                    {'name': 'Weights GiB', 'type': 'line', 'data': [], 'smooth': False, 'showSymbol': False, 'lineStyle': {'type': 'dashed'}},
                    {'name': 'KV Used GiB', 'type': 'line', 'data': [], 'smooth': True, 'showSymbol': False, 'areaStyle': {}},
                ],
                'animation': False,
            }).classes('w-1/2 h-64')

        # Numeric stats
        ui.label('Latency & Counters').classes('text-subtitle2 font-bold')
        with ui.row().classes('gap-4'):
            with ui.card().classes('p-3'):
                ui.label('TTFT (avg)').classes('text-caption')
                ttft_label = ui.label('--').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('ITL (avg)').classes('text-caption')
                itl_label = ui.label('--').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('E2E (avg)').classes('text-caption')
                e2e_label = ui.label('--').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('Preemptions').classes('text-caption')
                preempt_label = ui.label('0').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('Tokens Gen').classes('text-caption')
                tokens_label = ui.label('0').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('KV / Weight').classes('text-caption')
                kv_ratio_label = ui.label('--').classes('text-h6')
            with ui.card().classes('p-3'):
                ui.label('KV bytes/token').classes('text-caption')
                kv_bpt_label = ui.label('--').classes('text-h6')

    # Chart data buffers
    timestamps = deque(maxlen=MAX_POINTS)
    kv_data = deque(maxlen=MAX_POINTS)
    prefix_data = deque(maxlen=MAX_POINTS)
    tps_data = deque(maxlen=MAX_POINTS)
    running_data = deque(maxlen=MAX_POINTS)
    waiting_data = deque(maxlen=MAX_POINTS)
    gpu_util_data = deque(maxlen=MAX_POINTS)
    gpu_temp_data = deque(maxlen=MAX_POINTS)
    gpu_power_data = deque(maxlen=MAX_POINTS)
    weight_gib_data = deque(maxlen=MAX_POINTS)
    kv_used_gib_data = deque(maxlen=MAX_POINTS)

    async def refresh_server_list():
        running = vllm_service.list_running()
        opts = {port: f":{info['port']} — {info['model']}" for port, info in running.items()}
        server_select.options = opts
        server_select.update()
        if len(opts) == 1:
            server_select.value = next(iter(opts.keys()))
        elif opts and not server_select.value:
            server_select.value = next(iter(opts.keys()))

    async def poll_metrics():
        port = server_select.value
        if not port:
            return

        gpu = await run.io_bound(metrics_service.fetch_gpu_metrics)
        if gpu:
            used = gpu["gpu_mem_used_mb"] / 1024
            total = gpu["gpu_mem_total_mb"] / 1024
            gpu_mem_label.set_text(f"{used:.1f} / {total:.1f} GB")
            gpu_mem_bar.set_value(used / total if total else 0)
            gpu_util_label.set_text(f"{gpu['gpu_util_pct']:.0f}%")
            gpu_util_bar.set_value(gpu["gpu_util_pct"] / 100)
            gpu_temp_label.set_text(f"{gpu['gpu_temp_c']:.0f}C")
            gpu_power_label.set_text(f"{gpu['gpu_power_w']:.0f}W")

        cpu = await run.io_bound(metrics_service.fetch_cpu_metrics, port)
        if cpu:
            rss = cpu.get("cpu_mem_rss_mb", 0)
            peak = max(metrics_service.get_peak_rss_mb(port), rss)
            cpu_rss_label.set_text(f"{rss:,.0f} MB" if rss else "--")
            cpu_rss_peak_label.set_text(f"{peak:,.0f} MB" if peak else "--")
            host_used = cpu.get("host_mem_used_mb", 0)
            host_total = cpu.get("host_mem_total_mb", 0)
            if host_total:
                host_mem_label.set_text(f"{host_used/1024:.1f} / {host_total/1024:.1f} GB")
                host_mem_bar.set_value(host_used / host_total)

        vllm = await run.io_bound(metrics_service.fetch_vllm_metrics, port)
        if not vllm:
            return

        now = time.strftime("%H:%M:%S")
        timestamps.append(now)

        # GPU chart
        gpu_util_data.append(gpu.get("gpu_util_pct", 0) if gpu else 0)
        gpu_temp_data.append(gpu.get("gpu_temp_c", 0) if gpu else 0)
        gpu_power_data.append(gpu.get("gpu_power_w", 0) if gpu else 0)
        gpu_chart.options['xAxis']['data'] = list(timestamps)
        gpu_chart.options['series'][0]['data'] = list(gpu_util_data)
        gpu_chart.options['series'][1]['data'] = list(gpu_temp_data)
        gpu_chart.options['series'][2]['data'] = list(gpu_power_data)
        gpu_chart.update()
        kv_data.append(round(vllm.get("kv_cache_pct", 0), 1))
        prefix_data.append(round(vllm.get("prefix_cache_hit_rate", 0), 1))
        tps_data.append(round(vllm.get("gen_tokens_per_sec", 0), 1))
        running_data.append(vllm.get("requests_running", 0))
        waiting_data.append(vllm.get("requests_waiting", 0))

        cache_chart.options['xAxis']['data'] = list(timestamps)
        cache_chart.options['series'][0]['data'] = list(kv_data)
        cache_chart.options['series'][1]['data'] = list(prefix_data)
        cache_chart.update()

        throughput_chart.options['xAxis']['data'] = list(timestamps)
        throughput_chart.options['series'][0]['data'] = list(tps_data)
        throughput_chart.options['series'][1]['data'] = list(running_data)
        throughput_chart.options['series'][2]['data'] = list(waiting_data)
        throughput_chart.update()

        ttft = vllm.get("ttft_avg_ms", 0)
        ttft_label.set_text(f"{ttft:.1f} ms" if ttft > 0 else "--")
        itl = vllm.get("itl_avg_ms", 0)
        itl_label.set_text(f"{itl:.1f} ms" if itl > 0 else "--")
        e2e = vllm.get("e2e_avg_ms", 0)
        e2e_label.set_text(f"{e2e:.0f} ms" if e2e > 0 else "--")
        preempt_label.set_text(f"{int(vllm.get('preemptions', 0)):,}")
        tokens_label.set_text(f"{int(vllm.get('gen_tokens_total', 0)):,}")

        weight_gib = vllm.get("weight_mem_gib", 0) or 0
        kv_used_gib = vllm.get("kv_mem_used_gib", 0) or 0
        weight_gib_data.append(round(weight_gib, 2))
        kv_used_gib_data.append(round(kv_used_gib, 2))
        mem_chart.options['xAxis']['data'] = list(timestamps)
        mem_chart.options['series'][0]['data'] = list(weight_gib_data)
        mem_chart.options['series'][1]['data'] = list(kv_used_gib_data)
        mem_chart.update()

        ratio = vllm.get("kv_to_weight_ratio", 0)
        kv_ratio_label.set_text(f"{ratio:.2f}x" if ratio else "--")
        bpt = vllm.get("kv_bytes_per_token", 0)
        if bpt >= 1024 * 1024:
            kv_bpt_label.set_text(f"{bpt/(1024*1024):.1f} MB")
        elif bpt >= 1024:
            kv_bpt_label.set_text(f"{bpt/1024:.1f} KB")
        elif bpt > 0:
            kv_bpt_label.set_text(f"{bpt:.0f} B")
        else:
            kv_bpt_label.set_text("--")

    async def refresh_all():
        await refresh_server_list()
        await poll_metrics()

    refresh_btn.on_click(refresh_all)
    ui.timer(2.0, refresh_server_list)
    ui.timer(0.5, poll_metrics)

    return refresh_all
