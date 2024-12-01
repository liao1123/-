from scheme_AES import *
from collections import defaultdict
import torch
import time
import pickle
import os
import hyper_parameter as hp
from tqdm import tqdm


def pack_Search(query, R):
    # QueryGen
    start_token_generation = time.time()
    token = user.tokenGen(query, R, data_owner.Gen())
    end_token_generation = time.time()

    # Search
    start_search = time.time()
    res = user.hammingSearch(enIMI, token, R, partitions_dict)
    end_search = time.time()

    # Human verification
    reslen = sum([len(i) for i in res])

    res = {
        'Initialization': end_init - start_init,
        'Encryption': end_encryption - start_encryption,
        'Token generation': end_token_generation - start_token_generation,
        'Search': end_search - start_search,
        'Result number': reslen,
    }

    return res

R = hp.R  # Search radius
hash_len = hp.hash_len  # 3 Hashlen
ss = hp.ss  # Number of subcodes

blocksize = hp.blocksize
hashLenByte = hash_len >> 3 # 字节数

# 列表里面嵌套集合（键值对） 方便添加集合
IMI_image = [defaultdict(set) for i in range(ss)]
enIMI_image = [{} for i in range(ss)]
IMI_text = [defaultdict(set) for i in range(ss)]
enIMI_text = [{} for i in range(ss)]

# 初始化
start_init = time.time()

# 导入image text hash文件
hash_code_path = hp.hash_code_path
image_hash_path = os.path.join(hash_code_path, "image_hash_64")
text_hash_path = os.path.join(hash_code_path, "text_hash_64")
assert image_hash_path, "image hash not exist!!"
assert text_hash_path, "text hash not exist!!"
image_data = torch.load(image_hash_path)  # [{"image_id":0, "image_hash_64":0101001}] hash值是字符串形式保存的
text_data = torch.load(text_hash_path)  # [{"text_id":0, "text_hash_64":0101001}]

# img text num
image_hash, text_hash = [], []
for data in image_data:
    image_hash.append(data['image_hash_64'])
for data in text_data:
    text_hash.append(data['text_hash_64'])
image_num = len(image_hash)
text_num = len(text_hash)
print(f"image hash num : {image_num}")
print(f"text hash num : {text_num}")

# 导入IMI索引文件和对应数据库原文件，不存在就制作文件
IMI_path = hp.IMI_path
IMI_image_file = os.path.join(IMI_path, f'IMI_image_{hash_len}_{ss}_.pkl')
IMI_text_file = os.path.join(IMI_path, f'IMI_text_{hash_len}_{ss}_.pkl')
if os.path.exists(IMI_image_file) and os.path.exists(IMI_text_file):
    with open(IMI_image_file, 'rb') as f:
        IMI_image = pickle.load(f)
    with open(IMI_text_file, 'rb') as f:
        IMI_text = pickle.load(f)
else:
    interval = (hashLenByte // ss)

    # image hash 制作IMI
    for i in tqdm(range(image_num)):
        content = int(image_hash[i], 2)  # 转化成十进制int
        content = content.to_bytes(hashLenByte, byteorder='big')  # 把content转化成与hashLenByte一样大小的字节，"big"表示大端表示法
        for j, II in enumerate(IMI_image):
            l, r = interval * j, interval * (j + 1)
            II[content[l: r]].add(i)

    with open(IMI_image_file, 'wb') as f:
        pickle.dump(IMI_image, f)

    # text hash 制作IMI
    for i in tqdm(range(text_num)):
        content = int(text_hash[i], 2)  # 转化成十进制int
        content = content.to_bytes(hashLenByte, byteorder='big')  # 把content转化成与hashLenByte一样大小的字节，"big"表示大端表示法
        for j, II in enumerate(IMI_text):
            l, r = interval * j, interval * (j + 1)
            II[content[l: r]].add(i)

    with open(IMI_text_file, 'wb') as f:
        pickle.dump(IMI_text, f)

# IMI每个M的value的最大数目，因为最终加密的时候要进行padding
max_img_ElemNum = []
max_text_ElemNum = []
for idx, II in enumerate(IMI_image):
    max_img_ElemNum.append(max([len(item) for item in II.values()]))
for idx, II in enumerate(IMI_text):
    max_text_ElemNum.append(max([len(item) for item in II.values()]))

# 代表的是原始数据
m = {i: f'msg{i}' for i in range(image_num)}
v = [f'' for i in range(image_num)]
M = (m, v)
maxLenV = max([len(i) for i in v])

# data owner 对IMI 进行加密操作的一个类
data_owner = Label(lenV=maxLenV, blocksize=blocksize)
# 代表search user的一个类，基本上是搜索方案
user = User(blocksize=blocksize, lenV=maxLenV, hashLen=hash_len, K=data_owner.Gen(), ss=ss)

partitions_dict = {}

# 采用递归的方法去给出指定R下所有SS可能的r情况，比如nn=2, ss=2，输出结果为[0,0] [0,1] [1,0] [1,1]
for nn in range(R + 1):
    partitions = partition_ordered(nn, ss, hash_len // ss)
    partitions_dict[(nn, ss)] = partitions
end_init = time.time()

# IndexBuld
enIMI_path = hp.enIMI_path
enIMI_image_file = os.path.join(enIMI_path, f'enIMI_image_{hash_len}_{ss}.pkl')
enIMI_text_file = os.path.join(enIMI_path, f'enIMI_text_{hash_len}_{ss}.pkl')
# # 导入原始数据
# c_file = os.path.join(folder, f'c_{db}_{hashLen}.pkl')

if os.path.exists(enIMI_image_file) and os.path.exists(enIMI_text_file):
    with open(enIMI_image_file, 'rb') as f:
        enIMI_image = pickle.load(f)
    with open(enIMI_text_file, 'rb') as f:
        enIMI_text = pickle.load(f)
    # with open(c_file, 'rb') as f:
    #     c = pickle.load(f)
else:
    start_encryption = time.time()
    # 加密IMI_image
    for i, II in enumerate(IMI_image):
        enIMI_image[i] = data_owner.Enc(data_owner.Gen(), II)
    # 加密IMI_text
    for i, II in enumerate(IMI_text):
        enIMI_text[i] = data_owner.Enc(data_owner.Gen(), II)
    end_encryption = time.time()

    with open(enIMI_image_file, 'wb') as f:
        pickle.dump(enIMI_image, f)
    with open(enIMI_text_file, 'wb') as f:
        pickle.dump(enIMI_text, f)

    # c = data_owner.EncData(M)
    # with open(c_file, 'wb') as f:
    #     pickle.dump(c, f)

