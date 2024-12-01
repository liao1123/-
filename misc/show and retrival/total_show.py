import gradio as gr
from utils import *
from misc.total_retrial import img2img_retrial, img2text_retrial, text2img_retrial


# 图搜图
def img2img_gr():
    examples = [
        ["https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/test2014/COCO_test2014_000000000069.jpg", 10,
         clip_small[0]],
        ["https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/test2014/COCO_test2014_000000000080.jpg", 10,
         clip_small[0]],
        ["https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/train2014/COCO_train2014_000000000009.jpg",
         10, clip_small[0]],
        ["https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/train2014/COCO_train2014_000000000308.jpg",
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


# 图搜文
def img2text_gr():
    examples = [
        ["https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/test2014/COCO_test2014_000000000069.jpg", 10,
         clip_small[0]],
        ["https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/test2014/COCO_test2014_000000000080.jpg", 10,
         clip_small[0]],
        ["https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/train2014/COCO_train2014_000000000009.jpg",
         10, clip_small[0]],
        ["https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/train2014/COCO_train2014_000000000308.jpg",
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


# 文搜图
def text2img_gr():
    examples = [
        ["手机壳", 10, clip_small[0]],
        ["陶瓷", 10, clip_small[0]],
        ["面包", 10, clip_small[0]],
        ["大树", 10, clip_small[0]]
    ]

    title = "<h1 align='center'>文本到图像检索demo</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    text = gr.Textbox(value="小猫", label="请填写文本", elem_id=0, interactive=True)
                num = gr.components.Slider(minimum=0, maximum=10, step=1, value=10, label="返回图片数（可能被过滤部分）",
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
    gr.close_all()
    #   多tag场景
    with gr.TabbedInterface(
            [text2img_gr(), img2img_gr(), img2text_gr()],
            ["文到图搜索", "图到图搜索", "图到文搜索"],
    ) as demo:
        demo.launch(share=True)
