import os

hash_len = 32 # 64
# 加密超参
R = 4  # Search radius
ss = 4  # Number of subcodes
blocksize = 16

# 训练MLP epoch
epoch = 20
seed = 2024

# dataset = "MUGE"
dataset = "Flickr"

# 一些路径
all_pretrain_model = ['rn50', 'vit-b-16']
model_name = all_pretrain_model[0]
all_vision_model = ["RN50", "ViT-B-16"]
vision_model = all_vision_model[0]
all_text_model = ["RBT3-chinese", "RoBERTa-wwm-ext-base-chinese"]
text_model = all_text_model[0]

IMI_path = os.path.join(f"datapath/IMI/{dataset}/{model_name}")
os.makedirs(IMI_path, exist_ok=True)

hash_code_path = os.path.join(f"datapath/hash_code/{dataset}/{model_name}")
os.makedirs(hash_code_path, exist_ok=True)

enIMI_path = os.path.join(f"datapath/enIMI/{dataset}/{model_name}")
os.makedirs(enIMI_path, exist_ok=True)

dataset_wash_path = f"datapath/datasets/dataset_wash/{dataset}"
os.makedirs(dataset_wash_path, exist_ok=True)

