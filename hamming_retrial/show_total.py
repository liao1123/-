import gradio as gr
from show_text2img import text2img_gr
from show_img2img import img2img_gr
from show_img2text import img2text_gr

if __name__ == "__main__":
    gr.close_all()
    #   多tag场景
    with gr.TabbedInterface(
            [text2img_gr(), img2img_gr(), img2text_gr()],
            ["文到图搜索", "图到图搜索", "图到文搜索"],
    ) as demo:
        demo.launch(share=True)
