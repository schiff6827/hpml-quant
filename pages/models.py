import time
from nicegui import ui, app, run
from services import hf_service
import config

COL_TO_FIELD = {
    'params': 'params_raw',
    'size': 'size_raw',
}
ZERO_BOTTOM_FIELDS = {'params_raw', 'size_raw', 'downloads_30d', 'downloads_all', 'trending', 'likes'}
EMPTY_BOTTOM_FIELDS = {'type', 'date'}
DESC_FIRST_COLS = {'params', 'size', 'downloads_30d', 'downloads_all', 'trending', 'likes', 'downloaded'}


def _sort_rows(rows, sort_by, descending):
    if not sort_by:
        return rows
    field = COL_TO_FIELD.get(sort_by, sort_by)
    def is_empty(val):
        if field in ZERO_BOTTOM_FIELDS:
            return val == 0 or val is None
        if field in EMPTY_BOTTOM_FIELDS:
            return val == '' or val is None
        if field == 'downloaded':
            return False
        return val is None or val == ''
    non_empty = []
    empties = []
    for row in rows:
        if is_empty(row.get(field)):
            empties.append(row)
        else:
            non_empty.append(row)
    non_empty.sort(key=lambda r: r.get(field, ''), reverse=descending)
    return non_empty + empties


def search_panel():
    with ui.row().classes('items-center gap-2 w-full flex-wrap'):
        provider_select = ui.select(
            hf_service.TOP_PROVIDERS,
            value="All",
            label='Provider',
            with_input=True,
        ).classes('w-40')
        search_input = ui.input('Search', placeholder='model name...').classes('w-64')
        search_btn = ui.button('Search', icon='search')
        clear_btn = ui.button('Clear', icon='clear').props('flat')
        dl_btn = ui.button('Download', icon='download').props('disable')
        dl_progress = ui.linear_progress(value=0, show_value=False).props('instant-feedback').classes('w-48')
        dl_progress.visible = False
        dl_status = ui.label('').classes('text-sm text-grey')
        dl_status.visible = False

    _TCELL = 'white-space: nowrap; overflow: hidden; text-overflow: ellipsis'

    results_table = ui.table(
        columns=[
            {'name': 'downloaded', 'label': 'DL', 'field': 'downloaded',
             'style': f'width: 36px; {_TCELL}', 'headerStyle': 'width: 36px'},
            {'name': 'provider', 'label': 'Provider', 'field': 'provider', 'align': 'left',
             'style': f'width: 100px; {_TCELL}', 'headerStyle': 'width: 100px'},
            {'name': 'model_name', 'label': 'Model', 'field': 'model_name', 'align': 'left',
             'style': f'width: 240px; max-width: 300px; {_TCELL}', 'headerStyle': 'width: 240px'},
            {'name': 'type', 'label': 'Type', 'field': 'type',
             'style': f'width: 90px; {_TCELL}', 'headerStyle': 'width: 90px'},
            {'name': 'params', 'label': 'Params', 'field': 'params_raw',
             'style': f'width: 70px; {_TCELL}', 'headerStyle': 'width: 70px'},
            {'name': 'size', 'label': 'Size', 'field': 'size_raw',
             'style': f'width: 80px; {_TCELL}', 'headerStyle': 'width: 80px'},
            {'name': 'downloads_30d', 'label': 'DL 30d', 'field': 'downloads_30d',
             'style': f'width: 80px; {_TCELL}', 'headerStyle': 'width: 80px'},
            {'name': 'downloads_all', 'label': 'DL All', 'field': 'downloads_all',
             'style': f'width: 80px; {_TCELL}', 'headerStyle': 'width: 80px'},
            {'name': 'trending', 'label': 'Trend', 'field': 'trending',
             'style': f'width: 55px; {_TCELL}', 'headerStyle': 'width: 55px'},
            {'name': 'likes', 'label': 'Likes', 'field': 'likes',
             'style': f'width: 60px; {_TCELL}', 'headerStyle': 'width: 60px'},
            {'name': 'gated', 'label': 'Gated', 'field': 'gated',
             'style': f'width: 50px; {_TCELL}', 'headerStyle': 'width: 50px'},
            {'name': 'date', 'label': 'Date', 'field': 'date',
             'style': f'width: 80px; {_TCELL}', 'headerStyle': 'width: 80px'},
        ],
        rows=[],
        row_key='id',
        selection='single',
        pagination={'rowsPerPage': 50},
    ).props('dense').classes('w-full')

    # Custom sortable headers — no Quasar sort, fully Python-controlled
    results_table.add_slot('header-cell', r'''
        <q-th :props="props" @click="() => $parent.$emit('sort_click', props.col.name)"
               class="cursor-pointer select-none" :style="props.col.headerStyle">
            {{ props.col.label }}
            <q-icon v-if="props.col.name === $parent.$props.sortCol"
                    :name="$parent.$props.sortDesc ? 'arrow_downward' : 'arrow_upward'" size="xs" class="q-ml-xs" />
        </q-th>
    ''')

    selected_row = {'current': None}
    all_rows = {'data': []}
    sort_state = {'col': None, 'desc': False}

    # Expose sort state to the Vue template via custom props
    results_table._props['sortCol'] = None
    results_table._props['sortDesc'] = False

    def apply_sort():
        sorted_rows = _sort_rows(all_rows['data'], sort_state['col'], sort_state['desc'])
        results_table.rows = sorted_rows
        results_table._props['sortCol'] = sort_state['col']
        results_table._props['sortDesc'] = sort_state['desc']
        results_table.update()

    def on_sort_click(e):
        col_name = e.args
        if col_name == sort_state['col']:
            sort_state['desc'] = not sort_state['desc']
        else:
            sort_state['col'] = col_name
            sort_state['desc'] = col_name in DESC_FIRST_COLS
        apply_sort()

    results_table.on('sort_click', on_sort_click)

    results_table.add_slot('body-cell-downloaded', '''
        <q-td :props="props">
            <q-icon v-if="props.row.downloaded" name="check_circle" color="green" size="xs" />
        </q-td>
    ''')
    results_table.add_slot('body-cell-provider', '''
        <q-td :props="props">
            <a :href="'https://huggingface.co/' + props.value" target="_blank"
               class="text-primary" style="text-decoration: none;">{{ props.value }}</a>
        </q-td>
    ''')
    results_table.add_slot('body-cell-model_name', '''
        <q-td :props="props" :title="props.row.id">
            <a :href="'https://huggingface.co/' + props.row.id" target="_blank"
               class="text-primary" style="text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; max-width: 300px;">{{ props.value }}</a>
        </q-td>
    ''')
    results_table.add_slot('body-cell-params', '''
        <q-td :props="props">{{ props.row.params }}</q-td>
    ''')
    results_table.add_slot('body-cell-size', '''
        <q-td :props="props">{{ props.row.size_gb }}</q-td>
    ''')
    results_table.add_slot('body-cell-downloads_30d', '''
        <q-td :props="props">{{ props.value?.toLocaleString() }}</q-td>
    ''')
    results_table.add_slot('body-cell-downloads_all', '''
        <q-td :props="props">{{ props.value?.toLocaleString() }}</q-td>
    ''')
    results_table.add_slot('body-cell-likes', '''
        <q-td :props="props">{{ props.value?.toLocaleString() }}</q-td>
    ''')
    results_table.add_slot('body-cell-trending', '''
        <q-td :props="props">{{ props.value?.toLocaleString() }}</q-td>
    ''')

    progress_state = {
        "value": 0.0, "bytes_downloaded": 0, "bytes_total": 0,
        "start_time": 0, "status": "",
    }

    def poll_progress():
        if not dl_progress.visible:
            return
        dl_progress.set_value(progress_state.get("value", 0))
        dl_bytes = progress_state.get("bytes_downloaded", 0)
        total_bytes = progress_state.get("bytes_total", 0)
        elapsed = time.time() - progress_state.get("start_time", time.time())
        if dl_bytes > 0 and total_bytes > 0:
            dl_gb = dl_bytes / 1e9
            total_gb = total_bytes / 1e9
            if elapsed > 2:
                rate = dl_bytes / elapsed
                remaining = (total_bytes - dl_bytes) / rate if rate > 0 else 0
                if remaining < 60:
                    eta_str = f"{int(remaining)}s"
                else:
                    eta_str = f"{int(remaining / 60)}m {int(remaining % 60)}s"
            else:
                eta_str = "calculating..."
            dl_status.set_text(f"{dl_gb:.1f} / {total_gb:.1f} GB — ETA: {eta_str}")
        elif progress_state.get("status"):
            dl_status.set_text(progress_state["status"])

    ui.timer(1.0, poll_progress)

    async def do_search():
        search_btn.disable()
        ui.notify('Searching...', type='info', timeout=2000)
        prov = provider_select.value
        if prov == "All":
            prov = ""
        results = await run.io_bound(
            hf_service.search_models,
            query=search_input.value,
            provider=prov,
        )
        all_rows['data'] = results
        sort_state['col'] = None
        sort_state['desc'] = False
        results_table.rows = results
        results_table._props['sortCol'] = None
        results_table._props['sortDesc'] = False
        results_table.update()
        search_btn.enable()
        ui.notify(f'Showing {len(results)} models', type='positive')

    def do_clear():
        provider_select.value = "All"
        search_input.value = ""
        all_rows['data'] = []
        results_table.rows = []
        results_table.update()
        selected_row['current'] = None
        dl_btn.props('disable')

    def on_select(e):
        if e.selection:
            selected_row['current'] = e.selection[0]
            dl_btn.props(remove='disable')
        else:
            selected_row['current'] = None
            dl_btn.props('disable')

    async def do_download():
        row = selected_row['current']
        if not row:
            ui.notify('Select a model first', type='warning')
            return
        token = app.storage.general.get('hf_token', '')
        if row['gated'] != 'No' and not token:
            ui.notify('This model is gated. Set your HF token in Settings first.', type='warning')
            return
        dl_btn.props('disable')
        dl_progress.visible = True
        dl_status.visible = True
        progress_state["value"] = 0.0
        progress_state["bytes_downloaded"] = 0
        progress_state["bytes_total"] = row.get("size_raw", 0)
        progress_state["start_time"] = time.time()
        progress_state["status"] = "Starting download..."
        progress_state["expected_bytes"] = row.get("size_raw", 0)
        dl_progress.set_value(0.0)
        dl_status.set_text("Starting download...")
        try:
            await run.io_bound(hf_service.download_model, row['id'], token, progress_state)
            ui.notify(f"Downloaded {row['id']}", type='positive')
            for r in all_rows['data']:
                if r['id'] == row['id']:
                    r['downloaded'] = True
            apply_sort()
        except Exception as e:
            ui.notify(f"Download failed: {e}", type='negative')
        finally:
            dl_btn.props(remove='disable')
            dl_progress.visible = False
            dl_status.visible = False
            progress_state["value"] = 0.0

    search_btn.on_click(do_search)
    search_input.on('keydown.enter', do_search)
    clear_btn.on_click(do_clear)
    results_table.on_select(on_select)
    dl_btn.on_click(do_download)


def cached_panel():
    with ui.column().classes('w-full gap-2'):
        with ui.row().classes('gap-2'):
            refresh_btn = ui.button('Refresh', icon='refresh')
            delete_btn = ui.button('Delete Selected', icon='delete').props('color=negative')

        cached_grid = ui.table(
            columns=[
                {'name': 'id', 'label': 'Model', 'field': 'id', 'align': 'left', 'sortable': True},
                {'name': 'size_gb', 'label': 'Size on Disk', 'field': 'size_gb', 'sortable': True},
                {'name': 'revisions', 'label': 'Revisions', 'field': 'revisions'},
                {'name': 'last_modified', 'label': 'Last Modified', 'field': 'last_modified', 'sortable': True},
            ],
            rows=[],
            row_key='id',
            selection='single',
        ).props('binary-state-sort').classes('w-full')

        async def refresh():
            models = await run.io_bound(hf_service.list_cached_models)
            cached_grid.rows = models
            cached_grid.update()

        async def delete_selected():
            if not cached_grid.selected:
                ui.notify('Select a model first', type='warning')
                return
            row = cached_grid.selected[0]
            with ui.dialog() as confirm, ui.card():
                ui.label(f"Delete {row['id']}? This cannot be undone.")
                with ui.row():
                    ui.button('Delete', on_click=lambda: confirm.submit(True)).props('color=negative')
                    ui.button('Cancel', on_click=lambda: confirm.submit(False)).props('flat')
            result = await confirm
            if result:
                await run.io_bound(hf_service.delete_cached_model, row['id'])
                ui.notify(f"Deleted {row['id']}", type='positive')
                await refresh()

        refresh_btn.on_click(refresh)
        delete_btn.on_click(delete_selected)
        ui.timer(0.1, refresh, once=True)

    return refresh
