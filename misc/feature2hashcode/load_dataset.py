import json

import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm

# # 遍历feature去寻找相似的组成训练pair
# class CustomDataSet(Dataset):
#     def __init__(self, images, texts, K=5, threshold=0.9):
#         self.img_text_pair = []
#         self.K = K
#         self.threshold = threshold
#
#         print("Calculating cosine similarities for images to texts...")
#         for image_feature in tqdm(images):
#             cos_values = dot(image_feature, texts.T) / (norm(image_feature) * norm(texts, axis=1))
#             # count = 0  # 初始化计数器
#             # for idx in range(len(cos_values)):
#             #     if cos_values[idx] > self.threshold:
#             #         self.img_text_pair.append((image_feature, texts[idx]))
#             #         count += 1
#             #         if count >= self.K:  # 达到 K 值就退出
#             #             break
#
#             # 排序太耗费时间
#             indices = np.argsort(cos_values)[-self.K:]  # 获取前K个相似文本的索引
#             for idx in indices:
#                 self.img_text_pair.append((image_feature, texts[idx]))
#             if len(self.img_text_pair) >= 100000:
#                 break
#
#         print("Calculating cosine similarities for texts to images...")
#         for text_feature in tqdm(texts):
#             cos_values = dot(text_feature, images.T) / (norm(text_feature) * norm(images, axis=1))
#             # count = 0  # 初始化计数器
#             # for idx in range(len(cos_values)):
#             #     if cos_values[idx] > self.threshold:
#             #         self.img_text_pair.append((images[idx], text_feature))
#             #         count += 1
#             #         if count >= self.K:  # 达到 K 值就退出
#             #             break
#
#             indices = np.argsort(cos_values)[-self.K:]  # 获取前K个相似图像的索引
#             for idx in indices:
#                 self.img_text_pair.append((images[idx], text_feature))
#             if len(self.img_text_pair) >= 200000:
#                 break
#
#     def __getitem__(self, index):
#         img = torch.tensor(self.img_text_pair[index][0], dtype=torch.float32)
#         text = torch.tensor(self.img_text_pair[index][1], dtype=torch.float32)
#         return img, text, index
#
#     def __len__(self):
#         return len(self.img_text_pair)

# 直接从原始text数据集中的img text对来进行操作
class CustomDataSet(Dataset):
    def __init__(self, images, texts, K=5, threshold=0.9):
        self.img_text_pair = []
        self.K = K
        self.threshold = threshold

        print("Calculating cosine similarities for images to texts...")
        for image_feature in tqdm(images):
            cos_values = dot(image_feature, texts.T) / (norm(image_feature) * norm(texts, axis=1))
            # count = 0  # 初始化计数器
            # for idx in range(len(cos_values)):
            #     if cos_values[idx] > self.threshold:
            #         self.img_text_pair.append((image_feature, texts[idx]))
            #         count += 1
            #         if count >= self.K:  # 达到 K 值就退出
            #             break

            # 排序太耗费时间
            indices = np.argsort(cos_values)[-self.K:]  # 获取前K个相似文本的索引
            for idx in indices:
                self.img_text_pair.append((image_feature, texts[idx]))
            if len(self.img_text_pair) >= 100000:
                break

        print("Calculating cosine similarities for texts to images...")
        for text_feature in tqdm(texts):
            cos_values = dot(text_feature, images.T) / (norm(text_feature) * norm(images, axis=1))
            # count = 0  # 初始化计数器
            # for idx in range(len(cos_values)):
            #     if cos_values[idx] > self.threshold:
            #         self.img_text_pair.append((images[idx], text_feature))
            #         count += 1
            #         if count >= self.K:  # 达到 K 值就退出
            #             break

            indices = np.argsort(cos_values)[-self.K:]  # 获取前K个相似图像的索引
            for idx in indices:
                self.img_text_pair.append((images[idx], text_feature))
            if len(self.img_text_pair) >= 200000:
                break

    def __getitem__(self, index):
        img = torch.tensor(self.img_text_pair[index][0], dtype=torch.float32)
        text = torch.tensor(self.img_text_pair[index][1], dtype=torch.float32)
        return img, text, index

    def __len__(self):
        return len(self.img_text_pair)

def load_dataset(batch_size, feature_path):
    # 导入之前提取的img text feature
    img_feature_path = os.path.join(feature_path, "image_feature.jsonl")
    text_feature_path = os.path.join(feature_path, "text_feature.jsonl")
    assert os.path.exists(img_feature_path), f"{img_feature_path} not existed"
    assert os.path.exists(text_feature_path), f"{text_feature_path} not existed"

    with open(img_feature_path, 'r') as img_file:
        img = []
        for line in tqdm(img_file):
            line_dict = json.loads(line)
            img.append(np.array(line_dict['feature'], dtype=np.float32))
        combined_imgs = np.array(img, dtype=np.float32)

    with open(text_feature_path, 'r') as text_file:
        text = []
        for line in tqdm(text_file):
            line_dict = json.loads(line)
            text.append(np.array(line_dict['feature'], dtype=np.float32))
        combined_texts = np.array(text, dtype=np.float32)

    img_feature_len = len(combined_imgs[0])
    text_feature_len = len(combined_texts[0])
    assert img_feature_len == text_feature_len, "img and text dim has question!!"
    print(f"img feature len:{img_feature_len} text feature len:{text_feature_len}")

    dataset = CustomDataSet(images=combined_imgs, texts=combined_texts, K=5, threshold=0.9)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=0)
    print("make dataset and dataloader succeddfully!!")

    return img_feature_len, dataloader
