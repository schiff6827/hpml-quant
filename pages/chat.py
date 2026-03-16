import asyncio
import html as html_lib
from openai import AsyncOpenAI
from nicegui import ui, run, app
from services import vllm_service


def content():
    """Chat tab. Returns a refresh callable."""
    with ui.column().classes('w-full gap-2'):
        with ui.row().classes('items-center gap-4 w-full'):
            server_select = ui.select([], label='Server', with_input=True).classes('w-64')
            refresh_btn = ui.button('', icon='refresh').props('flat dense')
            temp_slider = ui.slider(min=0, max=2, step=0.1, value=0.7).classes('w-32')
            ui.label().bind_text_from(temp_slider, 'value', backward=lambda v: f'Temp: {v:.1f}')
            max_tokens = ui.number('Max tokens', value=1024, min=1, max=16384, step=256).classes('w-28')
            clear_btn = ui.button('Clear', icon='delete_sweep').props('flat')

        with ui.expansion('System prompt', icon='settings').classes('w-full'):
            system_input = ui.textarea(value='You are a helpful assistant.').classes('w-full')

        chat_container = ui.column().classes('w-full gap-1').style('max-height: 600px; overflow-y: auto')

        with ui.row().classes('w-full items-end gap-2'):
            user_input = ui.textarea(placeholder='Type a message...').classes('flex-grow').props('autogrow rows=1 dense')
            send_btn = ui.button('Send', icon='send').props('color=primary')

    messages = []

    async def refresh_servers():
        running = vllm_service.list_running()
        opts = {port: f":{info['port']} — {info['model']}" for port, info in running.items()}
        server_select.options = opts
        server_select.update()
        if len(opts) == 1:
            server_select.value = next(iter(opts.keys()))
        elif opts and not server_select.value:
            server_select.value = next(iter(opts.keys()))

    def clear_chat():
        messages.clear()
        chat_container.clear()

    async def send_message():
        port = server_select.value
        if not port:
            ui.notify('No server selected. Launch one from the Servers tab.', type='warning')
            return
        text = user_input.value.strip()
        if not text:
            return

        user_input.value = ''
        send_btn.props('disable')

        with chat_container:
            ui.chat_message(text, name='You', sent=True)
        ui.run_javascript(f'document.getElementById("c{chat_container.id}").lastElementChild?.scrollIntoView({{behavior:"smooth"}})')

        messages.append({"role": "user", "content": text})

        api_messages = []
        sys_text = system_input.value.strip()
        if sys_text:
            api_messages.append({"role": "system", "content": sys_text})
        api_messages.extend(messages)

        response_html = None
        with chat_container:
            with ui.chat_message(name='Assistant', sent=False) as msg:
                response_html = ui.html('<span style="color: #999; font-style: italic">Thinking...</span>')

        client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

        full_response = ""
        try:
            running = vllm_service.list_running()
            model_name = running.get(port, {}).get('model', 'default')

            stream = await client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                temperature=temp_slider.value,
                max_tokens=int(max_tokens.value),
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    escaped = html_lib.escape(full_response).replace('\n', '<br>')
                    response_html.content = escaped
                    ui.run_javascript(f'document.getElementById("c{chat_container.id}").lastElementChild?.scrollIntoView({{behavior:"smooth"}})')
                    await asyncio.sleep(0)

            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            response_html.content = f'<span style="color: red">Error: {html_lib.escape(str(e))}</span>'

        send_btn.props(remove='disable')

    refresh_btn.on_click(refresh_servers)
    clear_btn.on_click(clear_chat)
    send_btn.on_click(send_message)
    user_input.on('keydown.enter.prevent', send_message)

    ui.timer(2.0, refresh_servers)

    return refresh_servers
