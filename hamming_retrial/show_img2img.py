import gradio as gr
from retrial_img2img import *
from utils import *


def img2img_gr():
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

    title = "<h1 align='center'>图像到图像检索demo</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    img = gr.components.Image(label="图片", type="pil", elem_id=1)
                num = gr.components.Slider(minimum=0, maximum=10, step=1, value=8, label="返回图片数（可能被过滤部分）",
                                           elem_id=2)
                model = gr.components.Radio(label="模型选择",
                                            choices=[clip_small[0], clip_base[0], clip_large[0], clip_large_336[0],
                                                     clip_high[0]],
                                            value=clip_small[0], elem_id=3)
                btn = gr.Button("搜索", )
            with gr.Column(scale=100):
                out = gr.Gallery(label="检索结果为：", columns=4)
        inputs = [img, num, model]
        btn.click(fn=img2img_retrial, inputs=inputs, outputs=out)
        gr.Examples(examples, inputs=inputs)
    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [img2img_gr()],
            ["图到图搜索"],
    ) as demo:
        demo.launch(
                    )
