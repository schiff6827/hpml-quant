from nicegui import ui, app, run
import config
from services import notify_service


def content():
    with ui.column().classes('w-full p-4 gap-4 max-w-xl'):
        ui.label('Settings').classes('text-h5')

        # HF Token
        ui.label('HuggingFace Token').classes('text-subtitle1 font-bold')
        ui.label('Required for gated models (Llama, Gemma, etc). Get yours at huggingface.co/settings/tokens').classes('text-sm text-grey')
        token_input = ui.input(
            'HF Token',
            password=True,
            password_toggle_button=True,
            value=app.storage.general.get('hf_token', ''),
        ).classes('w-full')

        def save_token():
            app.storage.general['hf_token'] = token_input.value
            ui.notify('Token saved', type='positive')

        ui.button('Save Token', icon='save', on_click=save_token)

        ui.separator()

        # Model cache info
        ui.label('Model Storage').classes('text-subtitle1 font-bold')
        ui.label(f'Cache directory: {config.MODEL_CACHE_DIR}').classes('text-sm font-mono')

        ui.separator()

        # Defaults
        ui.label('Default vLLM Settings').classes('text-subtitle1 font-bold')
        ui.label(f'GPU Memory Utilization: {config.DEFAULT_GPU_MEM_UTIL}').classes('text-sm')
        ui.label(f'DType: {config.DEFAULT_DTYPE}').classes('text-sm')
        ui.label(f'vLLM Port Start: {config.VLLM_PORT_START}').classes('text-sm')

        ui.separator()

        # ── Notifications ──
        ui.label('Notifications').classes('text-h6')
        ui.label('Used by the benchmark queue: heartbeats every 30min, immediate alerts on failure.').classes('text-sm text-grey')

        # Gmail
        ui.label('Gmail (SMTP)').classes('text-subtitle1 font-bold mt-2')
        ui.label('Requires a Google App Password (myaccount.google.com/apppasswords). Your regular password will not work once 2FA is enabled.').classes('text-sm text-grey')
        gmail_enabled = ui.checkbox(
            'Enable Gmail notifications',
            value=app.storage.general.get('notify_gmail_enabled', False),
        )
        gmail_user = ui.input(
            'Gmail address (from)',
            value=app.storage.general.get('notify_gmail_user', ''),
        ).classes('w-full')
        gmail_password = ui.input(
            'App password',
            password=True,
            password_toggle_button=True,
            value=app.storage.general.get('notify_gmail_password', ''),
        ).classes('w-full')
        gmail_to = ui.input(
            'Send to (comma-separated for multiple; leave blank to send to yourself)',
            value=app.storage.general.get('notify_gmail_to', ''),
        ).classes('w-full')

        gmail_status = ui.label('').classes('text-sm')

        def save_gmail():
            app.storage.general['notify_gmail_enabled'] = gmail_enabled.value
            app.storage.general['notify_gmail_user'] = gmail_user.value.strip()
            app.storage.general['notify_gmail_password'] = gmail_password.value
            app.storage.general['notify_gmail_to'] = gmail_to.value.strip()
            ui.notify('Gmail settings saved', type='positive')

        async def test_gmail():
            save_gmail()
            gmail_status.set_text('Sending test...')
            ok, err = await run.io_bound(notify_service.send_test, 'gmail')
            if ok:
                gmail_status.set_text('✓ Test sent — check your inbox')
                ui.notify('Gmail test sent', type='positive')
            else:
                gmail_status.set_text(f'✗ Failed: {err}')
                ui.notify(f'Gmail test failed: {err}', type='negative')

        with ui.row().classes('gap-2'):
            ui.button('Save', icon='save', on_click=save_gmail)
            ui.button('Send Test', icon='send', on_click=test_gmail).props('outline')
