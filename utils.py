import torch
import numpy as np
import random
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import normalize
import os
from numpy import dot
from numpy.linalg import norm

import torch.nn.functional as F

clip_small = ["中文CLIP(small)", "rn50"]  # 77M参数量
clip_base = ["中文CLIP(Base)", "vit-b-16"]  # 188M参数量
clip_large = ["中文CLIP(Large)", "vit-l-14"]  # 406M参数量
clip_large_336 = ["中文CLIP(Large,336分辨率)", "vit-l-14-336"]  # 407M参数量
clip_high = ["中文CLIP(High large)", "vit-h-14"]  # 958M参数量


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        # self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()

    # 先对x归一化，然后线性层-激活-dropout-线性层--tanh函数--归一化变成最终的hash值
    def _ff_block(self, x):
        x = normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        # feat = self.relu(self.fc2(feat))
        hid = self.tohash(self.dp(feat))
        # 输出值在[-1, 1]之间，归一化后得到hash值
        out = self.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output


class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        # self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()

    # 先对x归一化，然后线性层-激活-dropout-线性层--tanh函数--归一化变成最终的hash值
    def _ff_block(self, x):
        x = normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        # feat = self.relu(self.fc2(feat))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output


class ContrastiveLoss(nn.Module):
    def __init__(self, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.register_buffer("temperature", torch.tensor(temperature).to(device))

    def forward(self, emb_i, emb_j):
        # dim=1横行 dim=0纵行
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        # 计算相似度矩阵
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)
        batch_size = emb_i.size(0)  # 当前批量大小
        # 生成消极掩码，对角线为0，其余为1
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(emb_i.device)  # 动态计算
        sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs
        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


def save_checkpoints(self, epoch):
    if not os.path.exists(self.model_save_dir):
        os.makedirs(self.model_save_dir)
    file_name = "MUGE" + '_hash_' + str(self.nbits) + f"_epoch_{epoch}" + ".pt"
    ckp_path = os.path.join(self.model_save_dir, file_name)
    print(f"save MLP model dir: {ckp_path}")
    obj = {
        # 'FusionTransformer': self.FuseTrans.state_dict(),
        'ImageMlp': self.ImageMlp.state_dict(),
        'TextMlp': self.TextMlp.state_dict()
    }
    torch.save(obj, ckp_path)
    print('**********Save the hash model successfully.**********')
