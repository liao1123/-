# -*- coding: utf-8 -*-
"""
针对CLIP预训练好的model 去分别提取清洗好（id操作）的数据集的feature
"""

import os
import argparse
import logging
from pathlib import Path
import json

import lmdb
import torch
from tqdm import tqdm

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.eval.data import get_eval_img_dataset, get_eval_txt_dataset

import hyper_parameter as hp


def extract_feature():
    # 默认参数
    gpu = 0
    precision = "fp32"  # "fp32" "fp16"
    img_batch_size = 64
    text_batch_size = 64
    context_length = 52

    # 可选参数
    vision_model = hp.vision_model
    text_model = hp.text_model
    model_name = hp.model_name
    pretrained_model_weight_path = "datapath/pretrained_weights/clip_cn_{}.pt".format(model_name)

    print(f"choose vision model : {vision_model}")
    print(f"choose text model : {text_model}")
    print("load model checkpoint from {}.".format(pretrained_model_weight_path))

    torch.cuda.set_device(gpu)

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
    model = CLIP(**model_info).cuda(0)
    # print(model)
    convert_weights(model)

    if precision == "amp" or precision == "fp32":
        convert_models_to_fp32(model)
    if precision == "fp16":
        convert_weights(model)

    # 导入预训练权重
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
    print("loading the model successfully")

    # 创建组合的img text feature保存路径
    feat_output_path = f"datapath/feature/{hp.dataset}/{model_name}"
    os.makedirs(feat_output_path, exist_ok=True)
    img_feat_output_path = os.path.join(feat_output_path, "image_feature")
    text_feat_output_path = os.path.join(feat_output_path, "text_feature")
    assert not os.path.exists(img_feat_output_path), f"img feat outpath exist!!"
    assert not os.path.exists(text_feat_output_path), f"text feat outpath exist!!"

    # 用wash好的img text data提取feature保存到combined里面
    img_data_path = f"datapath/datasets/dataset_wash/{hp.dataset}/image_dataset"
    text_data_path = f"datapath/datasets/dataset_wash/{hp.dataset}/text_dataset"

    img_data = get_eval_img_dataset(img_data_path, img_batch_size, vision_model)
    text_data = get_eval_txt_dataset(text_data_path, max_txt_length=context_length, text_batch_size=text_batch_size)

    # 提取text feature
    print(f'Extracting combined text features...')
    text_feature_list = []
    write_cnt = 0
    model.eval()
    dataloader = text_data.dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader):
            text_ids, texts = batch
            texts = texts.cuda(gpu, non_blocking=True)
            text_features = model(None, texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                text_record = {"text_id": text_id, "feature": text_feature}
                text_feature_list.append(text_record)
                write_cnt += 1
    torch.save(text_feature_list, text_feat_output_path)
    print('{} text features are stored in {}'.format(write_cnt, text_feat_output_path))

    # 提取img feature
    print(f'Extracting combined image features...')
    img_feature_list = []
    write_cnt = 0
    model.eval()
    dataloader = img_data.dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_ids, images = batch
            images = images.cuda(gpu, non_blocking=True)
            image_features = model(images, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                image_record = {"image_id": image_id, "feature": image_feature}
                img_feature_list.append(image_record)
                write_cnt += 1
    torch.save(img_feature_list, img_feat_output_path)
    print('{} image features are stored in {}'.format(write_cnt, img_feat_output_path))

    print("Feature extraction completed!")


if __name__ == '__main__':
    extract_feature()
