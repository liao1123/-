# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation. (with single-GPU)
'''

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


def extract_feature():
    # 默认参数
    extract_image_feats = True
    extract_text_feats = True
    gpu = 0
    precision = "amp"  # "fp32" "fp16"
    img_batch_size = 32
    text_batch_size = 32
    context_length = 52
    # 可选参数
    all_vision_model = ["RN50", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14"]
    vision_model = all_vision_model[0]
    all_text_model = ["RBT3-chinese", "RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese"]
    text_model = all_text_model[0]
    dataset = "MUGE" # 或许加上别的
    data_type = "train"  # "test" "valid" "train"
    all_pretrain_model = ['rn50', 'vit-b-16', 'vit-l-14', "vit-l-14-336", "vit-h-14"]
    model_name = all_pretrain_model[0]

    pretrained_model_weight_path = "datapath/pretrained_weights/clip_cn_{}.pt".format(model_name)
    img_data_path = f"datapath/datasets/{dataset}/lmdb/{data_type}/imgs"
    text_data_path = f"datapath/datasets/{dataset}/{data_type}_texts.jsonl"
    feat_output_path = f"datapath/feature_datasets/{dataset}_feature/{model_name}/{data_type}"

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
    print("loading the model successfully")

    # 提取data
    img_data = get_eval_img_dataset(img_data_path, img_batch_size, vision_model)
    text_data = get_eval_txt_dataset(text_data_path, max_txt_length=context_length, text_batch_size=text_batch_size)

    # 创建保存路径
    if not os.path.exists(feat_output_path):
        os.makedirs(feat_output_path)

    # 提取text feature并存储到LMDB
    if extract_text_feats:
        print('Extracting text features...')
        text_feat_json_path = os.path.join(feat_output_path, "text_feature.jsonl")
        write_cnt = 0
        with open(text_feat_json_path, "w") as fout:
            model.eval()
            dataloader = text_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    text_ids, texts = batch
                    texts = texts.cuda(gpu, non_blocking=True)
                    text_features = model(None, texts)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"text_id": text_id, "feature": text_feature})))
                        write_cnt += 1
        print('{} text features are stored in {}'.format(write_cnt, text_feat_json_path))

        # 获取 JSONL 文件的大小
        json_file_size = os.path.getsize(text_feat_json_path)
        map_size = json_file_size * 2

        # 将 JSONL 文件导入到 LMDB 方便存储
        print('Converting text features to LMDB...')
        text_feat_lmdb_path = os.path.join(feat_output_path, "text_feature.mdb")
        env_text = lmdb.open(text_feat_lmdb_path, map_size=map_size)
        with env_text.begin(write=True) as txn:
            with open(text_feat_json_path, 'r') as fin:
                for line in tqdm(fin):
                    obj = json.loads(line.strip())
                    text_id = str(obj['text_id']).encode('utf-8')
                    text_feature = json.dumps(obj['feature']).encode('utf-8')
                    txn.put(text_id, text_feature)
        print(f'Text features are stored in {text_feat_lmdb_path}')

        # os.remove(text_feat_json_path)

    # 提取image feature并存储到JSONL
    if extract_image_feats:
        print('Extracting image features...')
        image_feat_json_path = os.path.join(feat_output_path, "image_feature.jsonl")
        write_cnt = 0
        with open(image_feat_json_path, "w") as fout:
            model.eval()
            dataloader = img_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    image_ids, images = batch
                    images = images.cuda(gpu, non_blocking=True)
                    image_features = model(images, None)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                        write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, image_feat_json_path))

        # 获取 JSONL 文件的大小
        json_file_size = os.path.getsize(image_feat_json_path)
        map_size = json_file_size * 2 # 预留额外空间

        # 将 JSONL 文件导入到 LMDB
        print('Converting image features to LMDB...')
        image_feat_lmdb_path = os.path.join(feat_output_path, "image_feature.mdb")
        env_img = lmdb.open(image_feat_lmdb_path, map_size=map_size)
        with env_img.begin(write=True) as txn:
            with open(image_feat_json_path, 'r') as fin:
                for line in tqdm(fin):
                    obj = json.loads(line.strip())
                    image_id = str(obj['image_id']).encode('utf-8')
                    image_feature = json.dumps(obj['feature']).encode('utf-8')
                    txn.put(image_id, image_feature)
        print(f'Image features are stored in {image_feat_lmdb_path}')

        # os.remove(image_feat_json_path)

    print("Feature extraction completed!")


if __name__ == '__main__':
    extract_feature()
