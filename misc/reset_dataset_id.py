# -*- coding: utf-8 -*-
"""
对原始排序后的dataset进行重新编制id并重新排序（原始id有些是空的）
"""
import json
import os
import torch


def combined_dataset():
    dataset_path = f"datapath/datasets/combined_no_reset_id"

    # 创建组合的img text 保存路径
    combined_datasets_output_path = f"datapath/datasets/combined_reset_id"
    if not os.path.exists(combined_datasets_output_path):
        os.makedirs(combined_datasets_output_path)
    combined_img_path = os.path.join(combined_datasets_output_path, "image_dataset")
    combined_text_path = os.path.join(combined_datasets_output_path, "text_dataset")

    text_data_list = []  # text data
    image_id_map = {}  # 用于映射原始 ID 到新的连续 ID text要用
    img_data_list = [] # img data

    img_path = os.path.join(dataset_path, "image_dataset")
    with open(img_path, "r", encoding="utf-8") as img_file:
        for line in img_file:
            read_img_data = json.loads(line)
            img_data = {
                "image_id": len(img_data_list),  # 设置连续的 ID
                "image_base64": read_img_data['image_base64']
            }
            image_id_map[read_img_data['image_id']] = len(img_data_list)  # 建立 ID 映射
            img_data_list.append(img_data)

    text_path = os.path.join(dataset_path, "text_dataset")
    with open(text_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            text_data = json.loads(line)
            # 替换 image_ids 中的原始 ID
            text_data["image_ids"] = [image_id_map.get(str(img_id), img_id) for img_id in
                                      text_data["image_ids"]]
            text_data["text_id"] = len(text_data_list)  # 设置连续的 text_id
            text_data_list.append(text_data)

    torch.save(img_data_list, combined_img_path)
    print(f"image has combined to {combined_img_path}")

    torch.save(text_data_list, combined_text_path)
    print(f"text has combined to {combined_text_path}")

    print("dataset combined successfully")


if __name__ == '__main__':
    combined_dataset()
