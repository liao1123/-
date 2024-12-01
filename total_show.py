import gradio as gr
from utils import *
from total_retrial import *
import psutil
import GPUtil
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Initialize data
x_cpu_data = []
ydata_cpu = []
start_time = time.time()

matplotlib.use("Agg")

def get_CPU_usage():
    cpu_usage = psutil.cpu_percent(interval=0.01)

    x_cpu_data.append(time.time() - start_time)
    ydata_cpu.append(cpu_usage)

    # Keep only the latest 100 data points
    xdata_recent = x_cpu_data[-50:]
    ydata_cpu_recent = ydata_cpu[-50:]

    # Return the data for plotting
    return pd.DataFrame({
        "time": xdata_recent,
        "CPU Usage": ydata_cpu_recent
    })


def text2img_gr():
    examples = [
        ["拿照相机拍照", 20, clip_small[0]],
        ["两个人在聊天", 20, clip_small[0]],
        ["一些人在聚餐", 20, clip_small[0]],
        ["成年人在沙发上睡觉", 20, clip_small[0]]
    ]

    title = "<h1 align='center'>文本到图像检索</h1>"

    with gr.Blocks() as demo:
        timer = gr.Timer(0.1)
        gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=1):
                    text = gr.Textbox(value="拿着相机拍照", label="请填写文本", elem_id=0, interactive=True)
                num = gr.components.Slider(minimum=0, maximum=50, step=1, value=20, label="返回图片数（可能被过滤部分）",
                                           elem_id=1)
                model = gr.components.Radio(label="模型选择",
                                            choices=[clip_small[0], clip_base[0]],
                                            value=clip_small[0], elem_id=2)
                btn = gr.Button("检索", )

            with gr.Column(scale=100):
                out = gr.Gallery(label="检索结果为：", columns=4)
            with gr.Column(scale=1):
                timer = gr.Timer(0.1)

                plot1 = gr.LinePlot(
                    get_CPU_usage,
                    x="time",
                    y="CPU Usage",
                    every=timer,
                    y_lim=[0, 100],
                    title="CPU Usage",
                    x_axis_labels_visible=False
                )
                other_total_time = gr.Textbox(label="导入模型以及计算耗费时间(s)", elem_id=3, interactive=True)
                token_total_time = gr.Textbox(label="生成token耗费时间(s)", elem_id=4, interactive=True)
                search_total_time = gr.Textbox(label="search耗费时间(s)", elem_id=5, interactive=True)

        inputs = [text, num, model]
        outputs = [out, other_total_time, token_total_time, search_total_time]
        # 前端需要处理的参数 text文本，retrial num， model名称
        btn.click(fn=text2img_retrial, inputs=inputs, outputs=outputs)
        gr.Examples(examples, inputs=inputs)
    return demo


def img2text_gr():
    examples = [
        ["examples/拿着相机拍照.webp", 10,
         clip_small[0]],
        ["examples/聊天.webp", 10,
         clip_small[0]],
        ["examples/聚餐.webp",
         10, clip_small[0]],
        ["examples/睡觉.webp",
         10, clip_small[0]]
    ]
    title = "<h1 align='center'>图像到文本检索</h1>"

    with gr.Blocks() as demo:
        timer = gr.Timer(0.5)
        gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=1):
                    img = gr.components.Image(label="图片", type="pil", elem_id=1)
                num = gr.components.Slider(minimum=0, maximum=10, step=1, value=10, label="返回文本数",
                                           elem_id=2)
                model = gr.components.Radio(label="模型选择",
                                            choices=[clip_small[0], clip_base[0]],
                                            value=clip_small[0], elem_id=3)
                btn = gr.Button("检索", )
            with gr.Column(scale=100):
                output = gr.Textbox(value="检索结果为")
            with gr.Column(scale=1):
                timer = gr.Timer(0.1)

                plot1 = gr.LinePlot(
                    get_CPU_usage,
                    x="time",
                    y="CPU Usage",
                    every=timer,
                    y_lim=[0, 100],
                    title="CPU Usage",
                    x_axis_labels_visible=False
                )
                other_total_time = gr.Textbox(label="导入模型以及计算耗费时间(s)", elem_id=3, interactive=True)
                token_total_time = gr.Textbox(label="生成token耗费时间(s)", elem_id=4, interactive=True)
                search_total_time = gr.Textbox(label="search耗费时间(s)", elem_id=5, interactive=True)
        inputs = [img, num, model]
        outputs = [output, other_total_time, token_total_time, search_total_time]
        btn.click(fn=img2text_retrial, inputs=inputs, outputs=outputs)
        gr.Examples(examples, inputs=inputs)
    return demo


# def img2img_gr():
#
#     examples = [
#         ["examples/拿着相机拍照.webp", 10,
#          clip_small[0]],
#         ["examples/聊天.webp", 10,
#          clip_small[0]],
#         ["examples/聚餐.webp",
#          10, clip_small[0]],
#         ["examples/睡觉.webp",
#          10, clip_small[0]]
#     ]
#
#     title = "<h1 align='center'>图像到图像检索</h1>"
#
#     with gr.Blocks() as demo:
#         timer = gr.Timer(0.5)
#         gr.Markdown(title)
#         with gr.Row():
#             with gr.Column(scale=1):
#                 with gr.Column(scale=2):
#                     img = gr.components.Image(label="图片", type="pil", elem_id=1)
#                 num = gr.components.Slider(minimum=0, maximum=50, step=1, value=20, label="返回图片数（可能被过滤部分）",
#                                            elem_id=2)
#                 model = gr.components.Radio(label="模型选择",
#                                             choices=[clip_small[0], clip_base[0]],
#                                             value=clip_small[0], elem_id=3)
#                 btn = gr.Button("检索", )
#             with gr.Column(scale=100):
#                 out = gr.Gallery(label="检索结果为：", columns=4)
#             with gr.Column(scale=1):
#                 timer = gr.Timer(0.1)
#
#                 plot1 = gr.LinePlot(
#                     get_CPU_usage,
#                     x="time",
#                     y="CPU Usage",
#                     every=timer,
#                     y_lim=[0, 100],
#                     title="CPU Usage",
#                     x_axis_labels_visible=False
#                 )
#                 other_total_time = gr.Textbox(label="导入模型以及计算耗费时间(s)", elem_id=3, interactive=True)
#                 token_total_time = gr.Textbox(label="生成token耗费时间(s)", elem_id=4, interactive=True)
#                 search_total_time = gr.Textbox(label="search耗费时间(s)", elem_id=5, interactive=True)
#         inputs = [img, num, model]
#         outputs = [out, other_total_time, token_total_time, search_total_time]
#         btn.click(fn=img2img_retrial, inputs=inputs, outputs=outputs)
#         gr.Examples(examples, inputs=inputs)
#     return demo


if __name__ == "__main__":
    gr.close_all()

    # with gr.TabbedInterface(
    #         [text2img_gr(), img2img_gr(), img2text_gr()],
    #         ["文到图检索", "图到图检索", "图到文检索"]
    # ) as demo:
    #     demo.launch(server_port=100, share=True)

    with gr.TabbedInterface(
            [text2img_gr(), img2text_gr()],
            ["文到图检索", "图到文检索"]
    ) as demo:
        demo.launch(server_port=100, share=True)
