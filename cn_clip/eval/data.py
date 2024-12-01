import os
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import torch
import lmdb
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
import torchvision.datasets as datasets
from cn_clip.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class EvalTxtDataset(Dataset):
    def __init__(self, jsonl_filename, max_txt_length=24):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)

        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.texts = []
        text_data_list = torch.load(jsonl_filename)
        for text_data in text_data_list:
            self.texts.append((text_data['text_id'], text_data["text"]))

        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')

        self.max_txt_length = max_txt_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_id, text = self.texts[idx]
        text = tokenize([_preprocess_text(str(text))], context_length=self.max_txt_length)[0]
        return text_id, text


class EvalImgDataset(Dataset):
    def __init__(self, img_jsonl_path, resolution=224):
        assert os.path.exists(img_jsonl_path), "The image LMDB directory {} not exists!".format(img_jsonl_path)

        logging.debug(f'Loading image img_jsonl from {img_jsonl_path}.')

        self.images = []
        image_data_list = torch.load(img_jsonl_path)
        for img_data in image_data_list:
            self.images.append((img_data["image_id"], img_data['image_base64']))

        self.number_images = len(self.images)
        logging.info("The specified LMDB directory contains {} images.".format(self.number_images))

        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        return Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        img_id, image_b64 = self.images[idx]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # Decode base64 image
        image = self.transform(image)

        return img_id, image


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def get_eval_txt_dataset(text_data_path, max_txt_length=24, text_batch_size=32):
    text_jsonl_path = text_data_path
    dataset = EvalTxtDataset(
        text_jsonl_path,
        max_txt_length=max_txt_length)
    num_samples = len(dataset)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=text_batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


def get_eval_img_dataset(img_data_path, img_batch_size=32, vision_model="ViT-B-16", ):
    imgs_jsonl_path = img_data_path
    dataset = EvalImgDataset(
        imgs_jsonl_path, resolution=fetch_resolution(vision_model))
    num_samples = len(dataset)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=img_batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_zeroshot_dataset(args, preprocess_fn):
    dataset = datasets.ImageFolder(args.datapath, transform=preprocess_fn)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.img_batch_size,
        num_workers=args.num_workers,
        sampler=None,
    )

    return DataInfo(dataloader, None)