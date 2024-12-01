# -*- coding: utf-8 -*-
"""
把原始train test valid原始数据集合并一起
方便之后操作
这里对原始的id只进行了排序操作
不使用jsonl格式进行保存，读取太慢，虽然可以可视化，但是使用torch吧，读取快且省空间
"""
import json
import os
import torch


def combined_dataset():
    data_type_total = ["train", "valid", "test"]  # "test" "valid" "train"
    dataset_path = f"../datapath/datasets/MUGE_origin_dataset"

    # 创建组合的img text 保存路径
    combined_datasets_output_path = f"datapath/datasets/combined_no_reset_id"
    if not os.path.exists(combined_datasets_output_path):
        os.makedirs(combined_datasets_output_path)

    combined_img_path = os.path.join(combined_datasets_output_path, "image_dataset")
    combined_text_path = os.path.join(combined_datasets_output_path, "text_dataset")

    # 合并文本数据
    combined_texts = []
    for data_type in data_type_total:
        text_path = os.path.join(dataset_path, f"{data_type}_texts.jsonl")

        with open(text_path, "r", encoding="utf-8") as text_file:
            for line in text_file:
                combined_texts.append(json.loads(line))
        print(f"{data_type} text has combined into {combined_text_path}")

    # 原始数据中img-data的id是打乱的，下面需要进行重新排序
    img_data_list = []
    for data_type in data_type_total:
        img_path = os.path.join(dataset_path, f"{data_type}_imgs.tsv")

        with open(img_path, "r", encoding="utf-8") as img_file:
            for line in img_file:
                line = line.strip()
                if line:
                    img_id, base64_img = line.split("\t")
                    img_data = {
                        "image_id": img_id,
                        "image_base64": base64_img
                    }
                    img_data_list.append(img_data)
        print(f"{data_type} image has combined {combined_img_path}")
    # 按照 ID 对图片数据进行排序
    img_data_list.sort(key=lambda x: x["image_id"])

    # 保存合并的图像数据和文本数据为Torch文件
    torch.save(img_data_list, combined_img_path)
    torch.save(combined_texts, combined_text_path)
    print("dataset合并完成！")


if __name__ == '__main__':
    combined_dataset()
