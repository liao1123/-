import torch
import torch.nn.functional as F
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.register_buffer("temperature", torch.tensor(temperature).to(device))

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)
        batch_size = emb_i.size(0)  # 当前批量大小
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(emb_i.device)  # 动态计算
        sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs
        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
