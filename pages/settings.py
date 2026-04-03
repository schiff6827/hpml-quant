from nicegui import ui, app
import config


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
