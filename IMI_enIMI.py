from scheme_AES import *
from collections import defaultdict
import torch
import pickle
import os
import hyper_parameter as hp
from tqdm import tqdm


def make_IMI_enIMI():
    hash_len = hp.hash_len  # 3 Hashlen
    ss = hp.ss  # Number of subcodes
    blocksize = hp.blocksize
    hashLenByte = hash_len >> 3  # 字节数

    # 列表里面嵌套集合（键值对） 方便添加集合
    IMI_image = [defaultdict(set) for i in range(ss)]
    IMI_text = [defaultdict(set) for i in range(ss)]
    enIMI_image = [{} for i in range(ss)]
    enIMI_text = [{} for i in range(ss)]

    # 导入image text hash文件
    hash_code_path = hp.hash_code_path
    image_hash_path = os.path.join(hash_code_path, f"image_hash_{hash_len}")
    text_hash_path = os.path.join(hash_code_path, f"text_hash_{hash_len}")
    assert image_hash_path, "image hash not exist!!"
    assert text_hash_path, "text hash not exist!!"
    image_data = torch.load(image_hash_path)  # [{"image_id":0, "image_hash":0101001}] hash值是字符串形式保存的
    text_data = torch.load(text_hash_path)  # [{"text_id":0, "text_hash":0101001}]

    # img text num
    image_hash, text_hash = [], []
    for data in image_data:
        image_hash.append(data['image_hash'])
    for data in text_data:
        text_hash.append(data['text_hash'])
    print(image_hash[:1000])
    print(text_hash[:1000])
    image_num = len(image_hash)
    text_num = len(text_hash)
    print(f"image hash num : {image_num}")
    print(f"text hash num : {text_num}")

    # 制作IMI索引文件
    IMI_path = hp.IMI_path
    IMI_image_file = os.path.join(IMI_path, f'IMI_image_{hash_len}_{ss}_.pkl')
    IMI_text_file = os.path.join(IMI_path, f'IMI_text_{hash_len}_{ss}_.pkl')

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

    # data owner 对IMI 进行加密操作的一个类
    data_owner = Label(blocksize=blocksize)

    # 加密IMI
    enIMI_path = hp.enIMI_path
    enIMI_image_file = os.path.join(enIMI_path, f'enIMI_image_{hash_len}_{ss}.pkl')
    enIMI_text_file = os.path.join(enIMI_path, f'enIMI_text_{hash_len}_{ss}.pkl')

    for i, II in enumerate(IMI_image):
        enIMI_image[i] = data_owner.Enc(data_owner.Gen(), II)
    for i, II in enumerate(IMI_text):
        enIMI_text[i] = data_owner.Enc(data_owner.Gen(), II)

    with open(enIMI_image_file, 'wb') as f:
        pickle.dump(enIMI_image, f)
    with open(enIMI_text_file, 'wb') as f:
        pickle.dump(enIMI_text, f)


if __name__ == '__main__':
    make_IMI_enIMI()
