import gradio as gr
import os
import cv2
import numpy as np
from rembg import new_session, remove
from rembg.bg import download_models

# --- 1. KONFIGURASI PATH & PERSISTENSI MODEL ---
model_path = os.path.join(os.getcwd(), "models")
os.makedirs(model_path, exist_ok=True)
os.environ["U2NET_HOME"] = model_path

required_models = ("u2net", "isnet-general-use", "sam")

print(f"[*] Menyingkronkan model ke: {model_path}...")
try:
    download_models(required_models)
    print("[*] Status: Model siap digunakan.")
except Exception as e:
    print(f"[!] Gagal sinkronisasi model: {e}")

# --- 2. LOGIKA INFERENSI ---
def inference(file, model, x, y):
    if file is None: return None, None
    session = new_session(model)
    
    rembg_kwargs = {}
    if model == "sam":
        rembg_kwargs["sam_prompt"] = [{"type": "point", "data": [x, y], "label": 1}]
    
    mask = remove(
        file,
        session=session,
        only_mask=True,
        **rembg_kwargs
    )

    if mask.shape[:2] != file.shape[:2]:
        mask = cv2.resize(mask, (file.shape[1], file.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
    image = cv2.cvtColor(file, cv2.COLOR_BGR2BGRA)
    image[:, :, 3] = mask
    return (image, mask)

# --- 3. CSS KUSTOM (Style Hacker/Cyber) ---
custom_css = """
footer {display: none !important;}
.gradio-container {
    background-color: #09090b !important;
    color: #ffffff !important;
    font-family: 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', monospace !important;
}
h1 {
    color: #00ffff !important;
    text-transform: uppercase;
    letter-spacing: 0.3em;
    font-weight: 900 !important;
    font-style: italic;
    text-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
}
.cyber-panel {
    background-color: rgba(24, 24, 27, 0.6) !important;
    border: 2px solid rgba(0, 255, 255, 0.2) !important;
    border-radius: 2px !important;
    padding: 10px !important;
}
button.primary {
    background: #00ffff !important;
    color: #000000 !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    border: none !important;
    letter-spacing: 0.1em !important;
}
button.primary:hover {
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.5) !important;
}
"""

def get_coords(evt: gr.SelectData) -> tuple:
    return evt.index[0], evt.index[1]

def toggle_sam_ui(model: str):
    visible = model == "sam"
    return gr.update(visible=visible), gr.update(visible=visible)

# --- 4. PEMBANGUNAN DASHBOARD ---
# PERBAIKAN: Menghapus gr.Container() dan menggantinya dengan gr.Column()
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    with gr.Column(elem_id="main-container"): 
        # Menggunakan HTML untuk menyejajarkan Logo dan Teks Judul secara horizontal
        gr.HTML("""
            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
                <img src="https://raw.githubusercontent.com/omenxuinsgd/testcase-majore/refs/heads/main/MIT_Black.png" 
                     style="height: 60px; filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5));" 
                     alt="Logo Majore">
                <h1 style="font-size: 42px; margin: 0; padding: 0; border: none; text-shadow: 0 0 15px rgba(0, 255, 255, 0.4);">
                    MAJORE M-ONE AIO
                </h1>
            </div>
        """)
        gr.Markdown(f"""
        <span style="color: white; font-size: 18px; font-family: 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', monospace;">
            > HAPUS & RUBAH BACKGROUND || CORE_STATUS: ACTIVE || ENCRYPTION: ENABLED
        </span>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                inputs = gr.Image(type="numpy", label="SOURCE_BUFFER_INPUT", elem_classes="cyber-panel")
                
                model_selector = gr.Dropdown(
                    choices=list(required_models),
                    value="u2net",
                    label="VECTOR_MODEL_SELECTION"
                )
                
                with gr.Row(visible=False) as sam_controls:
                    x_coord = gr.Number(label="POINT_X", precision=0)
                    y_coord = gr.Number(label="POINT_Y", precision=0)
                
                process_btn = gr.Button("EXECUTE_DECODING", variant="primary", elem_classes="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="DECODED_RESULT", elem_classes="cyber-panel")
                output_mask = gr.Image(label="ALPHA_CHANNEL_MASK", elem_classes="cyber-panel")

        model_selector.change(toggle_sam_ui, inputs=model_selector, outputs=[sam_controls, sam_controls])
        inputs.select(get_coords, None, [x_coord, y_coord])

        process_btn.click(
            inference,
            inputs=[inputs, model_selector, x_coord, y_coord],
            outputs=[output_image, output_mask]
        )

        # gr.Examples(
        #     examples=[
        #         ["lion.png", "u2net", 0, 0],
        #         ["girl.jpg", "isnet-general-use", 0, 0]
        #     ],
        #     inputs=[inputs, model_selector, x_coord, y_coord]
        # )

if __name__ == "__main__":
    app.launch()