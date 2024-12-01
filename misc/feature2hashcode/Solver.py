import torch
import torch.optim as optim
from load_dataset import load_dataset
from metric import ContrastiveLoss
from model import ImageMlp, TextMlp
from utils import save_checkpoints
from torch.optim import lr_scheduler
from tqdm import tqdm


class Solver(object):
    def __init__(self, dataset, epoch, hash_lens, feature_path):
        self.dataset = dataset
        self.batch_size = 128
        self.total_epoch = epoch
        self.model_save_dir = "model_save"
        self.feature_path = feature_path

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")

        # 制作用于model训练的train的dataloader, 因为中文数据集没有搞到查询、检索来验证训练的好坏，直接就train看loss吧
        self.feat_lens, self.train_loader = load_dataset(self.batch_size, self.feature_path)
        self.nbits = hash_lens

        # 对CLIP输出的img text feature直接训练 MLP 来转化成二进制哈希码
        self.ImageMlp = ImageMlp(self.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TextMlp(self.feat_lens, self.nbits).to(self.device)

        # 打印model总共需要的参数数量
        paramsImage = list(self.ImageMlp.parameters())
        paramsText = list(self.TextMlp.parameters())
        total_param = sum([param.nelement() for param in paramsImage]) + sum([param.nelement() for param in paramsText])
        print("total_param:", total_param)

        # 优化器Adam
        self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))

        self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[30, 80], gamma=1.2)
        self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[30, 80], gamma=1.2)

        self.ContrastiveLoss = ContrastiveLoss(device=self.device)

    def train(self):
        print("Training Hash Fuction...")
        for epoch in range(self.total_epoch):
            train_loss = self.trainhash()
            print(f"epoch : {epoch + 1}, loss : {train_loss}")
        save_checkpoints(self)

    def trainhash(self):
        self.ImageMlp.train()
        self.TextMlp.train()
        running_loss = 0.0
        for idx, (img, txt, _) in enumerate(tqdm(self.train_loader)):
            img, txt = img.to(self.device), txt.to(self.device)

            img_embedding = self.ImageMlp(img)
            text_embedding = self.TextMlp(txt)
            loss = self.ContrastiveLoss(img_embedding, text_embedding)

            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            loss.backward()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            running_loss += loss.item()

            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()
        return running_loss


# if self.dataset == "mirflickr" or self.dataset == "nus-wide":
        #     self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[30, 80], gamma=1.2)
        #     self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[30, 80], gamma=1.2)
        # elif self.dataset == "mscoco":
        #     self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[200], gamma=0.6)
        #     self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[200], gamma=0.6)