import gradio as gr
from utils import *
from retrial_img2text import *


def img2text_gr():
    examples = [
        ["examples/手机壳.webp", 10,
         clip_small[0]],
        ["examples/手表.webp", 10,
         clip_small[0]],
        ["examples/帽子.webp",
         10, clip_small[0]],
        ["examples/牛仔裤.webp",
         10, clip_small[0]]
    ]

    title = "<h1 align='center'>图像到文本检索demo</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    img = gr.components.Image(label="图片", type="pil", elem_id=1)
                num = gr.components.Slider(minimum=0, maximum=10, step=1, value=10, label="返回文本数",
                                           elem_id=2)
                model = gr.components.Radio(label="模型选择",
                                            choices=[clip_small[0], clip_base[0], clip_large[0], clip_large_336[0],
                                                     clip_high[0]],
                                            value=clip_small[0], elem_id=3)
                btn = gr.Button("搜索", )
            with gr.Column(scale=100):
                output = gr.Textbox(value="检索结果为")
        inputs = [img, num, model]
        btn.click(fn=img2text_retrial, inputs=inputs, outputs=output)
        gr.Examples(examples, inputs=inputs)
    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [img2text_gr()],
            ["图到文搜索"],
    ) as demo:
        demo.launch(
                    )
