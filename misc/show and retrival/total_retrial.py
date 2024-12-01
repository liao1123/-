import base64
from io import BytesIO
from tqdm import tqdm
import json

import numpy as np
import torch
import os
from PIL import Image

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.clip import tokenize
from utils import *
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

input_image = Image.open('../../examples/pokemon.jpeg')

# 导入事先计算好的 img 特征从 img_feature_path
img_feature_path = "datapath/feature/rn50/image_feature.jsonl"
image_ids = []
image_feats = []

with open(img_feature_path, "r") as fin:
    for line in tqdm(fin):
        obj = json.loads(line.strip())
        image_ids.append(obj['image_id'])
        image_feats.append(obj['feature'])

image_feats_array = np.array(image_feats, dtype=np.float32)
print("Loading image features successfully.")

# 导入事先计算好的 text 特征从 text_feature_path
text_feature_path = "datapath/feature/rn50/text_feature.jsonl"
text_ids = []
text_feats = []

with open(text_feature_path, "r") as fin:
    for line in tqdm(fin):
        obj = json.loads(line.strip())
        text_ids.append(obj['text_id'])
        text_feats.append(obj['feature'])

text_feats_array = np.array(text_feats, dtype=np.float32)
print("Loading text features successfully.")


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


def _convert_to_rgb(img):
    return img.convert("RGB")


# 根据行数和id的关系更快读取, id 和 行数 是一样的
def get_image_list(img_path, img_id_list):
    img_data_list = []

    # 读取整个文件并存储在列表中
    with open(img_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 一次性读取所有行

    for img_id in img_id_list:
        line_number = img_id  # id 和 行数是一样的
        img_data = json.loads(lines[line_number].strip())
        b64_data = img_data['image_base64']
        img = base64.b64decode(b64_data)
        img = Image.open(BytesIO(img))
        img_data_list.append(img)

    return img_data_list


def get_text_list(text_path, text_id_list):
    print(f"text path: {text_path}")
    print(f"text id list: {text_id_list}")
    text_data_list = []

    # 读取整个文件并存储在列表中
    with open(text_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 一次性读取所有行

    for text_id in text_id_list:
        line_number = text_id
        entry = json.loads(lines[line_number].strip())
        text_data_list.append(entry.get('text'))

    print(text_data_list)
    # 单个文本框输出
    formatted_str = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(text_data_list)])
    print(f"text: {formatted_str}")
    return formatted_str


def text2img_retrial(input_text='艺术石膏', top_k_num=10, model_name=clip_small[0]):
    if model_name == "中文CLIP(Base)":
        model_name = "vit-b-16"
        vision_model = "ViT-B-16"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(Large)":
        model_name = "vit-l-14"
        vision_model = "ViT-L-14"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(Large,336分辨率)":
        model_name = "vit-l-14-336"
        vision_model = "ViT-L-14-336"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(small)":
        model_name = "rn50"
        vision_model = "RN50"
        text_model = 'RBT3-chinese'
    elif model_name == "中文CLIP(High large)":
        model_name = 'vit-h-14'
        vision_model = "ViT-H-14"
        text_model = 'RoBERTa-wwm-ext-large-chinese'
    else:
        raise NameError

    pretrained_model_weight_path = "datapath/pretrained_weights/clip_cn_{}.pt".format(model_name)
    img_path = f"datapath/datasets/combined_reset_id/image_dataset.jsonl"
    precision = "amp"  # "fp32" "fp16"

    # 导入vision text model参数
    vision_model_config_file = f"cn_clip/clip/model_configs/{vision_model.replace('/', '-')}.json"
    assert os.path.exists(vision_model_config_file)
    text_model_config_file = f"cn_clip/clip/model_configs/{text_model.replace('/', '-')}.json"
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v

    # 初始化model参数
    model = CLIP(**model_info)
    convert_weights(model)

    if precision == "amp" or precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(0)
    if precision == "fp16":
        convert_weights(model)

    # 导入预训练权重
    print("load model checkpoint from {}.".format(pretrained_model_weight_path))
    assert os.path.exists(pretrained_model_weight_path), "The checkpoint file {} not exists!".format(
        pretrained_model_weight_path)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(0)
    checkpoint = torch.load(pretrained_model_weight_path, map_location='cpu')
    start_epoch = checkpoint["epoch"]
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    model.eval()
    print("loading the model successfully")

    print("Begin to compute top-{} predictions for texts...".format(top_k_num))
    # 文本预处理
    text = tokenize([_preprocess_text(str(input_text))], context_length=32)[0]
    text = text.unsqueeze(0)  # 增加一个batchsize维度
    text = text.cuda(0, non_blocking=True)
    text_feat_tensor = model(None, text)
    # 注意归一化
    text_feat_tensor /= text_feat_tensor.norm(dim=-1, keepdim=True)

    img_feats_tensor = torch.from_numpy(image_feats_array[:len(image_ids)]).cuda()  # [batch_size, feature_dim]
    batch_scores = text_feat_tensor @ img_feats_tensor.t()  # [1, batch_size]
    score_tuples = []
    for image_id, score in zip(image_ids[:len(image_ids)], batch_scores.squeeze(0).tolist()):
        score_tuples.append((image_id, score))
    top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:top_k_num]

    choose_image_ids = [entry[0] for entry in top_k_predictions]
    print(f"choose image ids:{choose_image_ids}")

    img_list = get_image_list(img_path=img_path, img_id_list=choose_image_ids)

    return img_list


def img2text_retrial(input_image=input_image, top_k_num=10, model_name=clip_small[0]):
    # 只用combined 把三种dataset合并到一起了，让数据集更大一点，不需要 "test" "valid" "train"
    if model_name == "中文CLIP(Base)":
        model_name = "vit-b-16"
        vision_model = "ViT-B-16"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(Large)":
        model_name = "vit-l-14"
        vision_model = "ViT-L-14"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(Large,336分辨率)":
        model_name = "vit-l-14-336"
        vision_model = "ViT-L-14-336"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(small)":
        model_name = "rn50"
        vision_model = "RN50"
        text_model = 'RBT3-chinese'
    elif model_name == "中文CLIP(High large)":
        model_name = 'vit-h-14'
        vision_model = "ViT-H-14"
        text_model = 'RoBERTa-wwm-ext-large-chinese'
    else:
        raise NameError

    pretrained_model_weight_path = "datapath/pretrained_weights/clip_cn_{}.pt".format(model_name)
    text_path = f"datapath/datasets/combined_reset_id/text_dataset.jsonl"
    precision = "amp"  # "fp32" "fp16"

    # 导入vision text model参数
    vision_model_config_file = f"cn_clip/clip/model_configs/{vision_model.replace('/', '-')}.json"
    assert os.path.exists(vision_model_config_file)
    text_model_config_file = f"cn_clip/clip/model_configs/{text_model.replace('/', '-')}.json"
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v

    # 初始化model参数
    model = CLIP(**model_info)
    convert_weights(model)

    if precision == "amp" or precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(0)
    if precision == "fp16":
        convert_weights(model)

    # 导入预训练权重
    print("load model checkpoint from {}.".format(pretrained_model_weight_path))
    assert os.path.exists(pretrained_model_weight_path), "The checkpoint file {} not exists!".format(
        pretrained_model_weight_path)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(0)
    checkpoint = torch.load(pretrained_model_weight_path, map_location='cpu')
    start_epoch = checkpoint["epoch"]
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    model.eval()
    print("loading the model successfully")

    # img预处理
    print("calucate the img feature!!!")
    resolution = model_info['image_resolution']
    transform = Compose([
        Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    input_image = transform(input_image).unsqueeze(0)  # 增加一个batchsize维度
    input_image = input_image.cuda(0, non_blocking=True)

    # 计算输入图像特征
    with torch.no_grad():
        input_img_feature = model(input_image, None)  # 获取输入图像的特征
        input_img_feature /= input_img_feature.norm(dim=-1, keepdim=True)

    print("Begin to compute top-{} predictions for texts...".format(top_k_num))
    text_feats_tensor = torch.from_numpy(text_feats_array[:len(text_ids)]).cuda()  # [batch_size, feature_dim]
    batch_scores = input_img_feature @ text_feats_tensor.t()  # [1, batch_size]
    score_tuples = []
    for text_id, score in zip(text_ids[:len(text_ids)], batch_scores.squeeze(0).tolist()):
        score_tuples.append((text_id, score))
    top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:top_k_num]

    # 不需要把id变成字符串
    choose_text_ids = [entry[0] for entry in top_k_predictions]

    text_list = get_text_list(text_path=text_path, text_id_list=choose_text_ids)

    return text_list


def img2img_retrial(input_image=input_image, top_k_num=10, model_name=clip_small[0]):
    # 只用combined 把三种dataset合并到一起了，让数据集更大一点，不需要 "test" "valid" "train"
    if model_name == "中文CLIP(Base)":
        model_name = "vit-b-16"
        vision_model = "ViT-B-16"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(Large)":
        model_name = "vit-l-14"
        vision_model = "ViT-L-14"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(Large,336分辨率)":
        model_name = "vit-l-14-336"
        vision_model = "ViT-L-14-336"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
    elif model_name == "中文CLIP(small)":
        model_name = "rn50"
        vision_model = "RN50"
        text_model = 'RBT3-chinese'
    elif model_name == "中文CLIP(High large)":
        model_name = 'vit-h-14'
        vision_model = "ViT-H-14"
        text_model = 'RoBERTa-wwm-ext-large-chinese'
    else:
        raise NameError

    pretrained_model_weight_path = "datapath/pretrained_weights/clip_cn_{}.pt".format(model_name)
    img_path = f"datapath/datasets/combined_reset_id/image_dataset.jsonl"
    precision = "amp"  # "fp32" "fp16"

    # 导入vision text model参数
    vision_model_config_file = f"cn_clip/clip/model_configs/{vision_model.replace('/', '-')}.json"
    assert os.path.exists(vision_model_config_file)
    text_model_config_file = f"cn_clip/clip/model_configs/{text_model.replace('/', '-')}.json"
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v

    # 初始化model参数
    model = CLIP(**model_info)
    convert_weights(model)

    if precision == "amp" or precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(0)
    if precision == "fp16":
        convert_weights(model)

    # 导入预训练权重
    print("load model checkpoint from {}.".format(pretrained_model_weight_path))
    assert os.path.exists(pretrained_model_weight_path), "The checkpoint file {} not exists!".format(
        pretrained_model_weight_path)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(0)
    checkpoint = torch.load(pretrained_model_weight_path, map_location='cpu')
    start_epoch = checkpoint["epoch"]
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    model.eval()
    print("loading the model successfully")

    # img预处理
    print("calucate the img feature!!!")
    resolution = model_info['image_resolution']
    transform = Compose([
        Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    input_image = transform(input_image).unsqueeze(0)  # 增加一个batchsize维度
    input_image = input_image.cuda(0, non_blocking=True)

    # 计算输入图像特征
    with torch.no_grad():
        input_img_feature = model(input_image, None)  # 获取输入图像的特征
        input_img_feature /= input_img_feature.norm(dim=-1, keepdim=True)

    print("Begin to compute top-{} predictions for texts...".format(top_k_num))
    img_feats_tensor = torch.from_numpy(image_feats_array[:len(image_ids)]).cuda()  # [batch_size, feature_dim]
    batch_scores = input_img_feature @ img_feats_tensor.t()  # [1, batch_size]
    score_tuples = []
    for image_id, score in zip(image_ids[:len(image_ids)], batch_scores.squeeze(0).tolist()):
        score_tuples.append((image_id, score))
    top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:top_k_num]

    choose_image_ids = [entry[0] for entry in top_k_predictions]
    print(f"choose image ids:{choose_image_ids}")

    img_list = get_image_list(img_path=img_path, img_id_list=choose_image_ids)

    return img_list

# # 遍历数据集读取id
# def get_text_list(text_path, text_id_list):
#     print(f"text path:{text_path}")
#     print(f"text id list :{text_id_list}")
#     text_data_list = []
#
#     # 打开 JSONL 文件
#     with open(text_path, 'r', encoding='utf-8') as file:
#         # 遍历文件中的每一行
#         for line in file:
#             entry = json.loads(line.strip())
#             if entry.get('text_id') in text_id_list:
#                 text_data_list.append(entry.get('text'))
#     print(text_data_list)
#     # 单个文本框输出
#     formatted_str = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(text_data_list)])
#     print(f"text :{formatted_str}")
#     return formatted_str


# def get_image_list(img_path, img_id_list):
#     img_data_list = []  # 用于保存解码后的图像
#     with open(img_path, 'r', encoding='utf-8') as file:
#         for line in tqdm(file):
#             # 解析每一行 json
#             img_data = json.loads(line)
#             img_id = img_data['image_id']
#
#             # 检查当前 img_id 是否在 img_id_list 中
#             if img_id in img_id_list:
#                 # 解码 Base64 图像数据
#                 b64_data = img_data['image_base64']
#                 img = base64.b64decode(b64_data)
#                 img = Image.open(BytesIO(img))
#                 img_data_list.append(img)
#
#     return img_data_list