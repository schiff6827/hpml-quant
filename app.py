import config
from nicegui import ui, app, background_tasks
import pages.models
import pages.servers
import pages.settings


@ui.page('/')
def index():
    with ui.header(elevated=True).classes('items-center bg-blue-900 text-white px-4 gap-4'):
        ui.label('HPML Model Manager').classes('text-h6 font-bold')
        ui.space()
        ui.link('Settings', '/settings').classes('text-white no-underline hover:underline')

    with ui.column().classes('w-full p-4 gap-4'):
        with ui.tabs().classes('w-full') as tabs:
            search_tab = ui.tab('Search HuggingFace')
            cached_tab = ui.tab('Downloaded Models')
            servers_tab = ui.tab('Servers')

        with ui.tab_panels(tabs, value=search_tab).classes('w-full'):
            with ui.tab_panel(search_tab):
                pages.models.search_panel()
            with ui.tab_panel(cached_tab):
                cached_refresh = pages.models.cached_panel()
            with ui.tab_panel(servers_tab):
                servers_refresh = pages.servers.content()

        def on_tab_change(e):
            if e.value == cached_tab:
                background_tasks.create(cached_refresh())
            elif e.value == servers_tab:
                background_tasks.create(servers_refresh())

        tabs.on_value_change(on_tab_change)


@ui.page('/settings')
def settings_page():
    with ui.header(elevated=True).classes('items-center bg-blue-900 text-white px-4 gap-4'):
        ui.label('HPML Model Manager').classes('text-h6 font-bold')
        ui.space()
        ui.link('Back', '/').classes('text-white no-underline hover:underline')
    pages.settings.content()


if __name__ == '__main__':
    ui.run(
        host=config.APP_HOST,
        port=config.APP_PORT,
        title='HPML Model Manager',
        reload=False,
        show=False,
        storage_secret=config.STORAGE_SECRET,
    )
