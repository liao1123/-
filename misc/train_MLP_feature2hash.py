import sys
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

from utils import setup_seed, TextMlp, ImageMlp, ContrastiveLoss, save_checkpoints, FuseTransEncoder


# 直接从原始text数据集中的img text对来进行操作
class CustomDataSet(Dataset):
    def __init__(self, images_text_pair):
        self.img_text_pair = images_text_pair

    def __getitem__(self, index):
        img_feature = torch.tensor(self.img_text_pair[index][0], dtype=torch.float32)
        text_feature = torch.tensor(self.img_text_pair[index][1], dtype=torch.float32)
        return img_feature, text_feature, index

    def __len__(self):
        return len(self.img_text_pair)


def load_dataset(batch_size, feat_path):
    # 导入之前提取的img text feature
    img_text_pair = torch.load(feat_path)

    img_text_pair = img_text_pair
    img_feature_len = len(img_text_pair[0][0])
    text_feature_len = len(img_text_pair[0][1])
    assert img_feature_len == text_feature_len, "img and text dim has question!!"
    print(f"img feature len:{img_feature_len} text feature len:{text_feature_len}")

    dataset = CustomDataSet(img_text_pair)
    print(f"img text pair num : {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    print("make dataset and dataloader successfully!!")

    return img_feature_len, dataloader


class Solver(object):
    def __init__(self, epoch, hash_lens, feature_path, model_save_path):
        self.batch_size = 128
        self.total_epoch = epoch
        self.model_save_dir = model_save_path
        self.feature_path = feature_path
        self.nbits = hash_lens

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")

        # 制作用于model训练的train的dataloader, 因为中文数据集没有搞到查询、检索来验证训练的好坏，直接就train看loss吧
        self.feat_lens, self.train_loader = load_dataset(self.batch_size, self.feature_path)
        num_layers, self.token_size, nhead = 2, 2 * self.feat_lens, 4

        # 对CLIP输出的img text feature直接训练 MLP 来转化成二进制哈希码
        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead).to(self.device)
        self.ImageMlp = ImageMlp(self.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TextMlp(self.feat_lens, self.nbits).to(self.device)

        # 打印model总共需要的参数数量
        paramsFuse_to_update = list(self.FuseTrans.parameters())
        paramsImage = list(self.ImageMlp.parameters())
        paramsText = list(self.TextMlp.parameters())
        Fuse_param = sum([param.nelement() for param in paramsFuse_to_update])
        MLP_param = sum([param.nelement() for param in paramsImage]) + sum([param.nelement() for param in paramsText])
        print("Fuse param:", Fuse_param)
        print("MLP_param:", MLP_param)

        # 优化器SGD
        self.optimizer_FuseTrans = optim.Adam(paramsFuse_to_update, lr=1e-4, betas=(0.5, 0.999))
        # self.optimizer_ImageMlp = optim.SGD(paramsImage, lr=1e-1, momentum=0.9, weight_decay=1e-4)
        # self.optimizer_TextMlp = optim.SGD(paramsText, lr=1e-1, momentum=0.9, weight_decay=1e-4)
        self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-3,  betas=(0.5, 0.999))
        self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3,  betas=(0.5, 0.999))

        # self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[2, 4])
        # self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[2, 4])

        self.ContrastiveLoss = ContrastiveLoss(device=self.device)

    def train(self):
        print("Training Hash Fuction...")
        for epoch in range(self.total_epoch):
            train_loss = self.trainhash()
            print("epoch : {}, loss : {:.6f}, lr : {:.6f}".format(epoch + 1, train_loss, self.optimizer_ImageMlp.param_groups[0]['lr']))
        save_checkpoints(self)

    def trainhash(self):
        self.FuseTrans.train()
        self.ImageMlp.train()
        self.TextMlp.train()
        running_loss = 0.0
        total_batches = 0
        for idx, (img, txt, _) in enumerate(tqdm(self.train_loader)):
            img, txt = img.to(self.device), txt.to(self.device)
            temp_tokens = torch.concat((img, txt), dim=1)
            temp_tokens = temp_tokens.unsqueeze(0)
            img_embedding, text_embedding = self.FuseTrans(temp_tokens)
            loss1 = self.ContrastiveLoss(img_embedding, text_embedding)

            img_embedding = self.ImageMlp(img_embedding)
            text_embedding = self.TextMlp(text_embedding)
            loss2 = self.ContrastiveLoss(img_embedding, text_embedding)

            loss = loss1 + loss2 * 0.5
            self.optimizer_FuseTrans.zero_grad()
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            loss.backward()
            print(f"{idx} loss: {loss.item()}")
            self.optimizer_FuseTrans.step()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            running_loss += loss.item()
            total_batches += 1

            if idx % 1000 == 0:
                binary_hash = (img_embedding >= 0.0).float()
                binary_hash = binary_hash.squeeze().cpu().numpy()
                for data in binary_hash:
                    binary_str = ''.join(str(int(b)) for b in data)  # 转换为二进制字符串
                    print(binary_str)

            # # 在每个epoch结束后调用学习率变化
            # self.ImageMlp_scheduler.step()
            # self.TextMlp_scheduler.step()

        # 返回平均损失
        return running_loss / total_batches


if __name__ == '__main__':
    # setting seed
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed = 2024
    setup_seed(seed)

    all_pretrain_model = ['rn50', 'vit-b-16', 'vit-l-14', "vit-l-14-336", "vit-h-14"]
    model_name = all_pretrain_model[0]
    # MLP model 保存地址
    model_save_path = f"model_save/{model_name}"
    # 提前生成好的img text feature pair
    feature_path = f"datapath/img_text_feature_pair/{model_name}/img_text_feature_pair"
    assert os.path.exists(feature_path), f"{feature_path} not exists"
    print(f"feat path : {feature_path}")

    epoch = 30
    hash_lens = 64

    # 输出日志地址
    out_path = 'log'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"MUGE_seed={seed}_hash-len={hash_lens}")
    sys.stdout = open(out_path, 'w', encoding='utf-8')
    print(f"log_dir: {out_path}")

    start_time = time.time()
    task_out = str(hash_lens) + " bits" + "train hash"

    print('=============== {}--{}--Total epochs:{} ==============='.format("MUGE", task_out, epoch))

    print('...Training is beginning...')
    solver = Solver(epoch=epoch, hash_lens=hash_lens, feature_path=feature_path, model_save_path=model_save_path)
    print("init end!!!")
    solver.train()
    time_elapsed = time.time() - start_time
    print(f"train total time:{time_elapsed}")
