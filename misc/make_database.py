# -*- coding: utf-8 -*-
'''
把合并后的dataset和提取的feature合并成一个jsonl三元组制作database
其实可以不用，直接id feature一个文件检索id 一个id data文件返回data
'''
import json
import os


def make_database():
    dataset = "MUGE" # "Flickr30k-CN" "MUGE"
    model_name = "vit-b-16" # "vit-l-14" "vit-l-14-336" "vit-b-16"
    dataset_path = f"datapath/datasets/{dataset}/combined"
    feature_path = f"datapath/feature_datasets/{dataset}_feature/{model_name}/combined"
    database_outpath = f"datapath/database/{dataset}/{model_name}"

    if not os.path.exists(database_outpath):
        os.makedirs(database_outpath)
    text_out_path = os.path.join(database_outpath, "text_database.jsonl")
    img_out_path = os.path.join(database_outpath, "img_database.jsonl")
    if os.path.exists(text_out_path) and os.path.exists(img_out_path):
        print("out path exist!!!!")
        return
    open(text_out_path, "w").close()
    open(img_out_path, "w").close()

    #文本合并
    text_dataset_path = os.path.join(dataset_path, "text_dataset.jsonl")
    text_feature_path = os.path.join(feature_path, "text_feature.jsonl")
    with open(text_dataset_path, "r", encoding="utf-8") as dataset_file, open(text_feature_path, "r", encoding="utf-8") as feature_file, open(text_out_path, "w", encoding="utf-8") as out_file:
        for line1, line2 in zip(dataset_file, feature_file):
            line1_dict = json.loads(line1)
            line2_dict = json.loads(line2)

            text_record = json.dumps({
                "text_id": line1_dict["text_id"],
                "text": line1_dict['text'],
                "feature": line2_dict["feature"]
            })
            out_file.write(f"{text_record}\n")
    print("text database制作完成！")

    #img合并
    img_dataset_path = os.path.join(dataset_path, "image_dataset.jsonl")
    img_feature_path = os.path.join(feature_path, "image_feature.jsonl")
    with open(img_dataset_path, "r", encoding="utf-8") as dataset_file, open(img_feature_path, "r", encoding="utf-8") as feature_file, open(img_out_path, "w", encoding="utf-8") as out_file:
        for line1, line2 in zip(dataset_file, feature_file):
            line1_dict = json.loads(line1)
            line2_dict = json.loads(line2)

            img_record = json.dumps({
                "image_id": line1_dict["id"],
                "image_base64": line1_dict['image_base64'],
                "feature": line2_dict["feature"]
            })
            out_file.write(f"{img_record}\n")
    print("img database制作完成！")


if __name__ == '__main__':
    make_database()
