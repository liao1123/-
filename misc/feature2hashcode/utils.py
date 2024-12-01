import random

import torch
import numpy as np
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoints(self):
    if not os.path.exists(self.model_save_dir):
        os.makedirs(self.model_save_dir)
    file_name = self.dataset + '_hash_' + str(self.nbits) + ".pth"
    ckp_path = os.path.join(self.model_save_dir, file_name)
    print(f"save MLP model dir: {ckp_path}")
    obj = {
        'ImageMlp': self.ImageMlp.state_dict(),
        'TextMlp': self.TextMlp.state_dict()
    }
    torch.save(obj, ckp_path)
    print('**********Save the hash model successfully.**********')
