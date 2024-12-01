import torch
from utils import ImageMlp, TextMlp
import os
from tqdm import tqdm
import json
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time
import hyper_parameter as hp


class feature_Dataset(Dataset):
    def __init__(self, feature):
        self.feature = feature

    def __getitem__(self, item):
        return torch.tensor(self.feature[item], dtype=torch.float32)

    def __len__(self):
        return len(self.feature)


def extract_hash():
    # 超参数
    hash_len = hp.hash_len
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    # 导入feature 文件
    model_name = hp.model_name
    feature_path = f"datapath/feature/{hp.dataset}/{model_name}"
    img_feature_path = os.path.join(feature_path, "image_feature")
    text_feature_path = os.path.join(feature_path, "text_feature")
    assert os.path.exists(img_feature_path), f"{img_feature_path} not existed"
    assert os.path.exists(text_feature_path), f"{text_feature_path} not existed"

    print("load feature init")
    start_time = time.time()
    img_feature = torch.load(img_feature_path)
    text_feature = torch.load(text_feature_path)
    feature_load_time = time.time() - start_time
    print("load img text feature successfully!!!")
    print(f"load feature need time : {feature_load_time}")

    img_feature_list, text_feature_list = [], []
    for img_data in img_feature:
        img_feature_list.append(img_data['feature'])
    for text_data in text_feature:
        text_feature_list.append(text_data['feature'])

    img_feature_len = len(img_feature_list[0])
    text_feature_len = len(text_feature_list[0])
    assert img_feature_len == text_feature_len, "img and text dim has question!!"
    print(f"img feature len:{img_feature_len} text feature len:{text_feature_len}")

    # 制作dataset dataloader
    img_dataset = feature_Dataset(img_feature_list)
    text_dataset = feature_Dataset(text_feature_list)
    img_dataloader = DataLoader(img_dataset, shuffle=False, batch_size=256)
    text_dataloader = DataLoader(text_dataset, shuffle=False, batch_size=256)

    # 导入先前训练好的MLP model
    img_model = ImageMlp(img_feature_len, hash_len).to(device)
    text_model = TextMlp(text_feature_len, hash_len).to(device)

    file_name = "MUGE" + '_hash_' + str(hash_len) + "_epoch_20" + ".pt"
    model_state_path = f'model_save/{hp.dataset}/{model_name}'
    model_state_path = os.path.join(model_state_path, file_name)
    state = torch.load(model_state_path)
    img_model.load_state_dict(state['ImageMlp'])
    text_model.load_state_dict(state['TextMlp'])

    img_model.eval()
    text_model.eval()

    img_hash_code, text_hash_code = [], []

    for idx, feature in enumerate(tqdm(img_dataloader)):
        feature = feature.to(device)

        output = img_model(feature)
        binary_hash = (output >= 0.0).float()
        binary_hash = binary_hash.squeeze().cpu().numpy()
        for data in binary_hash:
            binary_str = ''.join(str(int(b)) for b in data)  # 转换为二进制字符串
            img_hash_code.append(binary_str)  # 按照int 保存会有问题，缺位的现象，还是按照字符串保存吧

    for idx, feature in enumerate(tqdm(text_dataloader)):
        feature = feature.to(device)

        output = text_model(feature)
        binary_hash = (output >= 0).float()
        binary_hash = binary_hash.squeeze().cpu().numpy()
        for data in binary_hash:
            binary_str = ''.join(str(int(b)) for b in data)  # 转换为二进制字符串
            text_hash_code.append(binary_str)  # 按照int 保存

    img_data_list, text_data_list = [], []
    for data in img_hash_code:
        img_recode = {"image_id": len(img_data_list), "image_hash": data}
        img_data_list.append(img_recode)
    for data in text_hash_code:
        text_recode = {"text_id": len(text_data_list), "text_hash": data}
        text_data_list.append(text_recode)

    hash_path = f'datapath/hash_code/{hp.dataset}/{model_name}'
    os.makedirs(hash_path, exist_ok=True)

    img_hash_path = os.path.join(hash_path, f"image_hash_{hash_len}")
    text_hash_path = os.path.join(hash_path, f"text_hash_{hash_len}")
    torch.save(img_data_list, img_hash_path)
    torch.save(text_data_list, text_hash_path)
    print("save img text hash end!!")


if __name__ == '__main__':
    extract_hash()
