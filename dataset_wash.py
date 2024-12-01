# -*- coding: utf-8 -*-
"""
把原始train test valid原始数据集合并一起
方便之后操作
这里对原始的id先进行排序操作 然后重新设置id让他们连续
不使用jsonl格式进行保存，读取太慢，虽然可以可视化，但是使用torch吧，读取快且省空间
"""
import json
import os
import torch
import hyper_parameter as hp


def combined_dataset():
    data_type_total = ["train", "valid", "test"]  # "test" "valid" "train"
    dataset_path = f"datapath/datasets/{hp.dataset}_origin_dataset"

    dataset_wash_output_path = hp.dataset_wash_path

    combined_img_path = os.path.join(dataset_wash_output_path, "image_dataset")
    combined_text_path = os.path.join(dataset_wash_output_path, "text_dataset")

    # 合并原本文本数据
    text_data_list = []
    for data_type in data_type_total:
        text_path = os.path.join(dataset_path, f"{data_type}_texts.jsonl")

        with open(text_path, "r", encoding="utf-8") as text_file:
            for line in text_file:
                text_data_list.append(json.loads(line))
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

    # 对id进行重设置
    img_new_data_list = []
    text_new_data_list = []
    image_id_map = {}  # 用于映射原始 ID 到新的连续 ID text要用

    for img_data in img_data_list:
        image_id_map[img_data['image_id']] = len(img_new_data_list)  # 建立 ID 映射
        img_data["image_id"] = len(img_new_data_list)
        img_new_data_list.append(img_data)

    for text_data in text_data_list:
        text_data["text_id"] = len(text_new_data_list)  # 设置连续的 text_id
        text_data["image_ids"] = [image_id_map.get(str(img_id), img_id) for img_id in
                                  text_data["image_ids"]]
        text_new_data_list.append(text_data)

    # 保存合并的图像数据和文本数据为Torch文件
    torch.save(img_new_data_list, combined_img_path)
    torch.save(text_new_data_list, combined_text_path)


if __name__ == '__main__':
    combined_dataset()
