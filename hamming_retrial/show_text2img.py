import gradio as gr
from utils import *
from retrial_text2img import *


def text2img_gr():
    examples = [
        ["手机壳", 10, clip_small[0]],
        ["手表", 10, clip_small[0]],
        ["牛仔裤", 10, clip_small[0]],
        ["帽子", 10, clip_small[0]]
    ]

    title = "<h1 align='center'>文本到图像检索demo</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    text = gr.Textbox(value="帽子", label="请填写文本", elem_id=0, interactive=True)
                num = gr.components.Slider(minimum=0, maximum=50, step=1, value=10, label="返回图片数（可能被过滤部分）",
                                           elem_id=2)
                model = gr.components.Radio(label="模型选择",
                                            choices=[clip_small[0], clip_base[0], clip_large[0], clip_large_336[0],
                                                     clip_high[0]],
                                            value=clip_small[0], elem_id=3)
                btn = gr.Button("搜索", )
            with gr.Column(scale=100):
                out = gr.Gallery(label="检索结果为：", columns=4)
        inputs = [text, num, model]
        # 前端需要处理的参数 text文本，retrial num， model名称
        btn.click(fn=text2img_retrial, inputs=inputs, outputs=out)
        gr.Examples(examples, inputs=inputs)
    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [text2img_gr()],
            ["文到图检索"],
    ) as demo:
        demo.launch()
