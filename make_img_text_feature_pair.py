"""
利用原始dataset 里面的text数据里面的text img对提取feature对组成训练集来训练MLP网络转化hash
"""
import os

import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import time
import hyper_parameter as hp


def make_img_text_pair():
    model_name = hp.model_name
    # 提前生成好的feature
    feat_path = f"datapath/feature/{hp.dataset}/{model_name}"
    # 原始数据集里面text数据集里面的text img id对
    text_data_path = f"datapath/datasets/dataset_wash/{hp.dataset}/text_dataset"
    assert os.path.exists(feat_path), f"{feat_path} not exists"
    assert os.path.exists(text_data_path), f"{text_data_path} not exists"

    # 导入之前提取的img text feature
    img_feature_path = os.path.join(feat_path, "image_feature")
    text_feature_path = os.path.join(feat_path, "text_feature")
    assert os.path.exists(img_feature_path), f"{img_feature_path} not existed"
    assert os.path.exists(text_feature_path), f"{text_feature_path} not existed"

    start_time = time.time()
    text_data_list = torch.load(text_data_path)
    print("read text data end")
    img_feature_list = torch.load(img_feature_path)
    print("read image feature end")
    text_feature_list = torch.load(text_feature_path)
    print("read text feature end")
    load_time = time.time() - start_time
    print(f"load time : {load_time}")

    # # 选取img去到text里面查找相似度最高的text组成pair
    # img_feature_np = np.array([data['feature'] for data in img_feature_list][:1000])  # 使用列表推导式
    # text_feature_np = np.array([data['feature'] for data in text_feature_list])
    # print("get text img pair!!")
    #
    # start_time = time.time()
    # img_norms = norm(img_feature_np, axis=1, keepdims=True)  # (N, 1)
    # text_norms = norm(text_feature_np, axis=1, keepdims=True)  # (M, 1)
    # similarity_matrix = dot(img_feature_np, text_feature_np.T) / (img_norms * text_norms.T)
    # calculate_similarity_time = time.time() - start_time
    # print(f"calculate similarity time : {calculate_similarity_time}")
    #
    # # 找到每个图像特征的最高相似度和对应的文本特征索引
    # best_text_indices = np.argmax(similarity_matrix, axis=1)
    # best_similarities = np.max(similarity_matrix, axis=1)
    # # 0.61847495 0.56702474 0.59348605 0.5878625  0.5790649  0.62449139
    # # 0.59937971 0.58276099 0.58322011 0.60393309 0.59590576 0.56349996
    # # 0.60371816 0.62065831 0.59856772 0.57525224 0.60981493 0.60506304
    # print(best_similarities)
    #
    # # 创建图像-文本对
    # img_text_pair = [(img_feature_np[i], text_feature_np[best_text_indices[i]], best_similarities[i]) for i in
    #                  range(len(img_feature_np))]

    # 直接按照text data里面的text和img 来组成pair 打印相似度
    img_text_pair = []
    print("get text img pair!!")
    similarity = []
    for text_data in text_data_list:
        text_id = text_data['text_id']
        image_ids = text_data['image_ids']

        if len(image_ids) == 0:
            continue

        text_feature = np.array(text_feature_list[text_id]['feature'], dtype=np.float32)

        for img_id in image_ids:
            img_feature = np.array(img_feature_list[img_id]['feature'], dtype=np.float32)
            img_text_pair.append((img_feature, text_feature))

        # 计算img text feature的相似度
        similarity.append(dot(img_feature, text_feature) / norm(img_feature) * norm(text_feature))
    # 0.5371484, 0.52259475, 0.5344845, 0.56850433, 0.56011635, 0.6048635, 0.5904433, 0.5049408, 0.50229007,
    # 0.5451221, 0.59335196, 0.5792193, 0.55978084, 0.558224, 0.57908535, 0.5390602, 0.5818084, 0.56197834,
    # 0.5556894, 0.56823903, 0.49583602, 0.5210632, 0.55575323, 0.5520002, 0.5201533, 0.5421286
    print(f"similarity : {similarity}")

    # 保存img text feature pair
    save_path = f"datapath/img_text_feature_pair/{hp.dataset}/{model_name}"
    os.makedirs(save_path, exist_ok=True)
    file_name = "img_text_feature_pair"
    save_path = os.path.join(save_path, file_name)
    torch.save(img_text_pair, save_path)


if __name__ == '__main__':
    make_img_text_pair()
