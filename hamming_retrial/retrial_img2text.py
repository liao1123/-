# -*- coding: utf-8 -*-
'''
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs text-to-image prediction file for evaluation.
'''

import argparse
import base64
from io import BytesIO

import lmdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import json

import numpy as np
import torch
import os
from PIL import Image

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from cn_clip.clip import tokenize

from utils import *


def hamming_distance(str1, str2):
    """计算汉明距离"""
    return sum(el1 != el2 for el1, el2 in zip(str1, str2))


def _convert_to_rgb(img):
    return img.convert("RGB")


def get_text_list(text_id_list):
    text_data_list = []
    for text_id in text_id_list:
        text_data = origin_text_database[text_id]
        text_data_list.append(text_data['text'])

    print(text_data_list)
    # 单个文本框输出
    formatted_str = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(text_data_list)])
    print(f"text: {formatted_str}")
    return formatted_str


input_image = Image.open('examples/pokemon.jpeg')
# 导入事先计算好的 text 特征从 text_feature_path
text_hash_data_path = "datapath/hash_code/rn50/text_hash_32"
text_hash_data = torch.load(text_hash_data_path)
text_id = []
text_hash = []
for data in text_hash_data:
    text_hash.append(data['text_hash'])
    text_id.append(data['text_id'])
print("Loading text features hash successfully.")

# 导入原始的text data
text_path = f"datapath/datasets/dataset_wash/text_dataset"
origin_text_database = torch.load(text_path)
print("Loading text database successfully.")


def img2text_retrial(input_image=input_image, top_k_num=10, model_name=clip_small[0]):
    hash_len = 32

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
    precision = "fp32"  # "fp32" "fp16"

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

    # 导入img MLP model
    img_model = ImageMlp(len(input_img_feature[0]), hash_len).cuda().type(torch.float32)
    file_name = "MUGE" + '_hash_' + str(hash_len) + "_epoch_20"+".pt"
    model_state_path = f'model_save/{model_name}'
    model_state_path = os.path.join(model_state_path, file_name)
    state = torch.load(model_state_path)
    img_model.load_state_dict(state['ImageMlp'])

    img_model.eval()
    get_img_hash = img_model(input_img_feature)
    binary_hash = (get_img_hash >= 0).float()
    binary_hash = binary_hash.squeeze().cpu().numpy()
    binary_str = ''.join(str(int(b)) for b in binary_hash)  # 转换为二进制字符串

    ham_dis = []
    for data in text_hash:
        ham_dis.append(hamming_distance(data, binary_str))
    ham_dis = np.array(ham_dis)
    # 获取最小汉明距离的索引
    choose_text_ids = np.argsort(ham_dis)[:top_k_num]
    print(f"choose text ids : {choose_text_ids}")

    text_list = get_text_list(text_id_list=choose_text_ids)

    return text_list
