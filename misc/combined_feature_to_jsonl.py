# -*- coding: utf-8 -*-
"""
把按照train test valid数据集提取到的特征进行合并
实际是之前提取特征是分开train valid test来提取的，为了最终检索精度好些，就把三个数据集合并在一起
参数有些变化，与主干code一致
"""

import os


def combined_feature_jsonl():
    dataset = "MUGE"
    data_type_total = ["train", "valid", "test"]  # "test" "valid" "train"
    model_name = "vit-l-14-336" # "vit-l-14" "vit-l-14-336" "vit-b-16"
    feat_output_path = f"datapath/feature_datasets/MUGE_feature/vit-b-16"

    # 创建组合的img text feature保存路径
    combined_feat_output_path = f"datapath/feature_datasets/MUGE_feature/vit-b-16/combined"
    if not os.path.exists(combined_feat_output_path):
        os.makedirs(combined_feat_output_path)
    combined_img_feat_output_path = os.path.join(combined_feat_output_path, "image_feature.jsonl")
    combined_text_feat_output_path = os.path.join(combined_feat_output_path, "text_feature.jsonl")
    if os.path.exists(combined_img_feat_output_path) and os.path.exists(combined_img_feat_output_path):
        print("combined path exist!!")
        return
    open(combined_img_feat_output_path, "w").close()
    open(combined_text_feat_output_path, "w").close()

    # 制作text database
    with open(combined_text_feat_output_path, "w") as combined_text_file:
        for data_type in data_type_total:
            text_feat_json_path = os.path.join(feat_output_path, f"{data_type}", "text_feature.jsonl")

            # 打开每个数据集的 text feature jsonl 文件，逐行读取并写入到 combined 文件中
            with open(text_feat_json_path, "r") as text_file:
                for line in text_file:
                    combined_text_file.write(line)
            print(f"{data_type} text features 已经合并到 {combined_text_feat_output_path}")

    # 合并 image features
    with open(combined_img_feat_output_path, "w") as combined_img_file:
        for data_type in data_type_total:
            img_feat_json_path = os.path.join(feat_output_path, f"{data_type}", "image_feature.jsonl")

            # 打开每个数据集的 image feature jsonl 文件，逐行读取并写入到 combined 文件中
            with open(img_feat_json_path, "r") as img_file:
                for line in img_file:
                    combined_img_file.write(line)
            print(f"{data_type} image features 已经合并到 {combined_img_feat_output_path}")

    print("特征合并完成！")


if __name__ == '__main__':
    combined_feature_jsonl()
