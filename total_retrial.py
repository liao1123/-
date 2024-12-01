import base64
import json
import os.path
import pickle
import time
from io import BytesIO

from PIL import Image

import hyper_parameter as hp
from cn_clip.clip import tokenize
from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from scheme_AES import *
from utils import *

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)  # Numpy module.
#     random.seed(seed)  # Python random module.
#     torch.use_deterministic_algorithms(True) # for pytorch >= 1.8
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.benchmark = False
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
# setup_seed(0)
blocksize = hp.blocksize
hash_len = hp.hash_len
ss = hp.ss
R = hp.R  # 搜索半径R
hashLenByte = hash_len >> 3

# 导入两个主要的类， data_owner用于对data IMI 加密， user用于检索操作
data_owner = Label(blocksize=blocksize)
user = User(blocksize=blocksize, hashLen=hash_len, K=data_owner.Gen(), ss=ss)

# 导入原始的img text dataset
data_path = hp.dataset_wash_path
image_dataset_path = os.path.join(data_path, "image_dataset")
text_dataset_path = os.path.join(data_path, "text_dataset")
origin_image_database = torch.load(image_dataset_path)
origin_text_database = torch.load(text_dataset_path)
# print("Loading image text origin dataset successfully.")

# 导入事先计算好的image enIMI
enIMI_image = []
all_pretrain_model = ['rn50', 'vit-b-16']
for model_name in all_pretrain_model:
    enIMI_path = os.path.join(f"datapath/enIMI/{hp.dataset}/{model_name}")
    enIMI_image_path = os.path.join(enIMI_path, f"enIMI_image_{hash_len}_{ss}.pkl")
    with open(enIMI_image_path, "rb") as file:
        enIMI_image.append(pickle.load(file))
    # print(f"Loading {model_name} enIMI image successfully.")

# 导入事先计算好的text enIMI
enIMI_text = []
all_pretrain_model = ['rn50', 'vit-b-16']
for model_name in all_pretrain_model:
    enIMI_path = os.path.join(f"datapath/enIMI/{hp.dataset}/{model_name}")
    enIMI_image_path = os.path.join(enIMI_path, f"enIMI_text_{hash_len}_{ss}.pkl")
    with open(enIMI_image_path, "rb") as file:
        enIMI_text.append(pickle.load(file))
    # print(f"Loading {model_name} enIMI text successfully.")

input_image = Image.open('examples/帽子.webp')

partitions_dict = {}
# 采用递归的方法去给出指定R下所有SS可能的r情况，比如nn=2, ss=2，输出结果为[0,0] [0,1] [1,0] [1,1]
for nn in range(R + 1):
    partitions = partition_ordered(nn, ss, hash_len // ss)
    partitions_dict[(nn, ss)] = partitions


# 输入加密IMI，查询和汉明距离R
def pack_Search(enIMI, query, R):
    # 生成查询令牌
    query = int(query, 2)
    query = query.to_bytes(hashLenByte, byteorder="big")
    token_start_time = time.time()
    token = user.tokenGen(query, R, data_owner.Gen())
    token_total_time = time.time() - token_start_time

    print("加密token：")
    for key, value_list in token[0].items():
        print(f"r={key}时")
        for pair in value_list:
            print(f"token : {pair[0].hex()}, {pair[1].hex()}")

    # 产生搜索结果
    search_start_time = time.time()
    res = user.hammingSearch(enIMI, token, R, partitions_dict)
    search_total_time = time.time() - search_start_time
    print(f"返回加密检索结果：")
    for item in res:
        if item != set():
            for idx in item:
                print(f"{idx.hex()}")

    # 对res解密
    result_list = []
    for data in res:
        if data:  # 检查集合是否非空
            for item in data:
                de_id = decrypt(item, 'd' * blocksize, blocksize)
                result_list.append(int.from_bytes(de_id, byteorder='big'))  # 解码字符串
    token_total_time = f"{token_total_time:.6f}"
    search_total_time = f"{search_total_time:.6f}"
    return result_list, token_total_time, search_total_time


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


def _convert_to_rgb(img):
    return img.convert("RGB")


def get_image_list(img_id_list):
    img_data_list = []

    for img_id in img_id_list:
        img_data = origin_image_database[img_id]
        b64_data = img_data['image_base64']
        img = base64.b64decode(b64_data)
        img = Image.open(BytesIO(img))
        img_data_list.append(img)

    return img_data_list


def get_text_list(text_id_list):
    text_data_list = []
    for text_id in text_id_list:
        text_data = origin_text_database[text_id]
        text_data_list.append(text_data['text'])

    print(text_data_list)
    # 单个文本框输出
    formatted_str = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(text_data_list)])
    print(f"text: {formatted_str}")
    return formatted_str


def choose_vision_text_model(model_name):
    if model_name == "中文CLIP(small)":
        model_name = "rn50"
        vision_model = "RN50"
        text_model = 'RBT3-chinese'
        idx = 0
    elif model_name == "中文CLIP(Base)":
        model_name = "vit-b-16"
        vision_model = "ViT-B-16"
        text_model = 'RoBERTa-wwm-ext-base-chinese'
        idx = 1
    else:
        raise NameError
    return model_name, vision_model, text_model, idx


def text2img_retrial(input_text='牛仔裤', top_k_num=20, model_name=clip_small[0]):
    # model calculate load time
    other_time = time.time()
    model_name, vision_model, text_model, idx = choose_vision_text_model(model_name)

    pretrained_model_weight_path = "datapath/pretrained_weights/clip_cn_{}.pt".format(model_name)
    precision = "fp32"  # "fp32" "fp16"

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
    model.eval()
    print("loading the model successfully")

    print("Begin to compute top-{} predictions for texts...".format(top_k_num))
    # 文本预处理
    text = tokenize([_preprocess_text(str(input_text))], context_length=32)[0]
    text = text.unsqueeze(0)  # 增加一个batchsize维度
    text = text.cuda(0, non_blocking=True)
    text_feat_tensor = model(None, text)
    # 注意归一化
    text_feat_tensor /= text_feat_tensor.norm(dim=-1, keepdim=True)

    # 导入text MLP model
    text_model = TextMlp(len(text_feat_tensor[0]), hash_len).cuda()
    file_name = "MUGE" + '_hash_' + str(hash_len) + "_epoch_20" + ".pt"
    model_state_path = f'model_save/{hp.dataset}/{model_name}'
    model_state_path = os.path.join(model_state_path, file_name)
    state = torch.load(model_state_path)
    text_model.load_state_dict(state['TextMlp'])

    # 得到二进制字符串
    text_model.eval()
    get_text_hash = text_model(text_feat_tensor)
    binary_hash = (get_text_hash >= 0).float()
    binary_hash = binary_hash.squeeze().cpu().numpy()
    binary_str = ''.join(str(int(b)) for b in binary_hash)  # 转换为二进制字符串

    other_total_time = time.time() - other_time
    other_total_time = f"{other_total_time:.6f}"
    print(f"导入模型推理等所需时间 : {other_total_time}")

    # 查询
    search_result, token_total_time, search_total_time  = pack_Search(enIMI_image[idx], binary_str, R)
    print(f"生成token所需时间 : {token_total_time}")
    print(f"检索所需时间 : {search_total_time}")
    print(f"检索结果数量 : {len(search_result)}")
    choose_image_ids = search_result[:top_k_num]
    print(f"检索图像ID:{choose_image_ids}")

    img_list = get_image_list(img_id_list=choose_image_ids)

    return img_list, other_total_time, token_total_time, search_total_time


def img2text_retrial(input_image=input_image, top_k_num=10, model_name=clip_small[0]):
    other_time = time.time()
    model_name, vision_model, text_model, idx = choose_vision_text_model(model_name)

    pretrained_model_weight_path = "datapath/pretrained_weights/clip_cn_{}.pt".format(model_name)
    precision = "fp32"  # "fp32" "fp16"

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
    model.eval()
    print("loading the model successfully")

    # img预处理
    print("calucate the img feature!!!")
    resolution = model_info['image_resolution']
    transform = Compose([
        Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    input_image = transform(input_image).unsqueeze(0)  # 增加一个batchsize维度
    input_image = input_image.cuda(0, non_blocking=True)

    # 计算输入图像特征
    with torch.no_grad():
        input_img_feature = model(input_image, None)  # 获取输入图像的特征
        input_img_feature /= input_img_feature.norm(dim=-1, keepdim=True)

    # 导入img MLP model
    img_model = ImageMlp(len(input_img_feature[0]), hash_len).cuda().type(torch.float32)
    file_name = "MUGE" + '_hash_' + str(hash_len) + "_epoch_20" + ".pt"
    model_state_path = f'model_save/{hp.dataset}/{model_name}'
    model_state_path = os.path.join(model_state_path, file_name)
    state = torch.load(model_state_path)
    img_model.load_state_dict(state['ImageMlp'])

    img_model.eval()
    get_img_hash = img_model(input_img_feature)
    binary_hash = (get_img_hash >= 0).float()
    binary_hash = binary_hash.squeeze().cpu().numpy()
    binary_str = ''.join(str(int(b)) for b in binary_hash)  # 转换为二进制字符串

    other_total_time = time.time() - other_time
    other_total_time = f"{other_total_time:.6f}"
    print(f"导入模型推理等所需时间 : {other_total_time}")

    # 查询
    search_result, token_total_time, search_total_time  = pack_Search(enIMI_text[idx], binary_str, R)
    print(f"生成token所需时间 : {token_total_time}")
    print(f"检索所需时间 : {search_total_time}")
    print(f"检索结果数量 : {len(search_result)}")
    choose_text_ids = search_result[:top_k_num]
    print(f"检索文本ID:{choose_text_ids}")

    text_list = get_text_list(text_id_list=choose_text_ids)

    return text_list, other_total_time, token_total_time, search_total_time


def img2img_retrial(input_image=input_image, top_k_num=10, model_name=clip_small[0]):
    othr_time = time.time()
    model_name, vision_model, text_model, idx = choose_vision_text_model(model_name)

    pretrained_model_weight_path = "datapath/pretrained_weights/clip_cn_{}.pt".format(model_name)
    precision = "fp32"  # "fp32" "fp16"

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
    model.eval()
    print("loading the model successfully")

    # img预处理
    print("calucate the img feature!!!")
    resolution = model_info['image_resolution']
    transform = Compose([
        Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    input_image = transform(input_image).unsqueeze(0)  # 增加一个batchsize维度
    input_image = input_image.cuda(0, non_blocking=True)

    # 计算输入图像特征
    with torch.no_grad():
        input_img_feature = model(input_image, None)  # 获取输入图像的特征
        input_img_feature /= input_img_feature.norm(dim=-1, keepdim=True)

        # 导入img MLP model
    img_model = ImageMlp(len(input_img_feature[0]), hash_len).cuda().type(torch.float32)
    file_name = "MUGE" + '_hash_' + str(hash_len) + "_epoch_20" + ".pt"
    model_state_path = f'model_save/{hp.dataset}/{model_name}'
    model_state_path = os.path.join(model_state_path, file_name)
    state = torch.load(model_state_path)
    img_model.load_state_dict(state['ImageMlp'])

    img_model.eval()
    get_img_hash = img_model(input_img_feature)
    binary_hash = (get_img_hash >= 0).float()
    binary_hash = binary_hash.squeeze().cpu().numpy()
    binary_str = ''.join(str(int(b)) for b in binary_hash)  # 转换为二进制字符串

    other_total_time = time.time() - othr_time
    other_total_time = f"{other_total_time:.6f}"
    print(f"导入模型推理等所需时间 : {other_total_time}")

    # 查询
    search_result, token_total_time, search_total_time = pack_Search(enIMI_image[idx], binary_str, R)
    print(f"生成token所需时间 : {token_total_time}")
    print(f"检索所需时间 : {search_total_time}")
    print(f"检索结果数量 : {len(search_result)}")
    choose_image_ids = search_result[:top_k_num]
    print(f"检索图像ID:{choose_image_ids}")

    img_list = get_image_list(img_id_list=choose_image_ids)

    return img_list, other_total_time, token_total_time, search_total_time


if __name__ == '__main__':
    text2img_retrial()