import config
from nicegui import ui, app, background_tasks
import pages.models
import pages.servers
import pages.monitor
import pages.chat
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
            monitor_tab = ui.tab('Monitor')
            chat_tab = ui.tab('Chat')

        with ui.tab_panels(tabs, value=search_tab).classes('w-full'):
            with ui.tab_panel(search_tab):
                pages.models.search_panel()
            with ui.tab_panel(cached_tab):
                cached_refresh = pages.models.cached_panel()
            with ui.tab_panel(servers_tab):
                servers_refresh = pages.servers.content()
            with ui.tab_panel(monitor_tab):
                monitor_refresh = pages.monitor.content()
            with ui.tab_panel(chat_tab):
                chat_refresh = pages.chat.content()

        def on_tab_change(e):
            if e.value == cached_tab:
                background_tasks.create(cached_refresh())
            elif e.value == servers_tab:
                background_tasks.create(servers_refresh())
            elif e.value == monitor_tab:
                background_tasks.create(monitor_refresh())
            elif e.value == chat_tab:
                background_tasks.create(chat_refresh())

        tabs.on_value_change(on_tab_change)


@ui.page('/settings')
def settings_page():
    with ui.header(elevated=True).classes('items-center bg-blue-900 text-white px-4 gap-4'):
        ui.label('HPML Model Manager').classes('text-h6 font-bold')
        ui.space()
        ui.link('Back', '/').classes('text-white no-underline hover:underline')
    pages.settings.content()


if __name__ == '__main__':
    from services import vllm_service
    vllm_service.reconnect_orphans()
    print(f'\n  HPML Model Manager: http://{config.APP_HOSTNAME}:{config.APP_PORT}\n')

    ui.run(
        host=config.APP_HOST,
        port=config.APP_PORT,
        title='HPML Model Manager',
        reload=False,
        show=False,
        storage_secret=config.STORAGE_SECRET,
    )
