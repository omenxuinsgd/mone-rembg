import gradio as gr
import os
import cv2
import numpy as np
from rembg import new_session, remove
from rembg.bg import download_models

def inference(file, model, x, y):
    session = new_session(model)
    
    mask = remove(
        file,
        session=session,
        **{ "sam_prompt": [{"type": "point", "data": [x, y], "label": 1}] },
        only_mask=True
    )

    print("Image Done")
    if mask.shape[:2] != file.shape[:2]:
        mask = cv2.resize(mask, (file.shape[1], file.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
    image = cv2.cvtColor(file, cv2.COLOR_BGR2BGRA)
    image[:, :, 3] = mask

    return (image, mask)

title = "RemBG"
description = "Gradio demo for **[RemBG](https://github.com/danielgatis/rembg)**. To use it, simply upload your image, select a model, click Process, and wait."
badge = """
    <div style="position: fixed; left: 50%; text-align: center;">
        <a href="https://github.com/danielgatis/rembg" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/RemBG-Github-blue" alt="RemBG Github" />
        </a>
    </div>
"""
def get_coords(evt: gr.SelectData) -> tuple:
    return evt.index[0], evt.index[1]

def show_coords(model: str):
    visible = model == "sam"
    return gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible)

download_models(tuple())

with gr.Blocks() as app:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        inputs = gr.Image(type="numpy", label="Input Image")
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            output_mask = gr.Image(label="Output Mask")
        
    model_selector = gr.Dropdown(
        [
            "u2net",
            "u2netp",
            "u2net_human_seg",
            "u2net_cloth_seg",
            "silueta",
            "isnet-general-use",
            "isnet-anime",
            "sam",
            "bria-rmbg",
            "birefnet-general",
            "birefnet-general-lite",
            "birefnet-portrait",
            "birefnet-dis",
            "birefnet-hrsod",
            "birefnet-cod",
            "birefnet-massive",
        ],
        value="isnet-general-use",
        label="Model Selection"
    )

    extra = gr.Markdown("## Click on the image to capture coordinates (for SAM model)", visible=False)

    x = gr.Number(label="Mouse X Coordinate", visible=False)
    y = gr.Number(label="Mouse Y Coordinate", visible=False)

    model_selector.change(show_coords, inputs=model_selector, outputs=[x, y, extra])
    inputs.select(get_coords, None, [x, y])


    gr.Button("Process Image").click(
        inference,
        inputs=[inputs, model_selector, x, y],
        outputs=(output_image, output_mask)
    )

    # gr.Examples(
    #     examples=[
    #         ["lion.png", "u2net", 100, None, None],
    #         ["girl.jpg", "u2net", 100, None, None],
    #         ["anime-girl.jpg", "isnet-anime", 100, None, None]
    #     ],
    #     inputs=[inputs, model_selector, x, y],
    #     outputs=(output_image, output_mask)
    # )
    # gr.HTML(badge)

app.launch(share=True)

