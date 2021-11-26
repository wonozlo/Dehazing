import numpy as np
import os
import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from utils.transforms.transforms import get_transform_params, get_transform

from glob import glob
from PIL import Image
from torch.utils.data import Dataset


class HazeDataset(Dataset):
    def __init__(self, data_path: str, mode: str = "train", crop_size: int = 256):
        super(HazeDataset, self).__init__()
        self.crop_size = crop_size
        self.mode = mode
        if mode not in ["train", "val", "test"]:
            raise NotImplementedError("Mode %s is invalid" % mode)

        self.pathA = os.path.join(data_path, mode + "/haze")
        self.imgsA = list(sorted(os.listdir(self.pathA)))
        if self.mode == "train":
            self.pathB = os.path.join(data_path, mode + '/gt')
            self.imgsB = list(sorted(os.listdir(self.pathB)))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with open(os.path.join(self.pathA, self.imgsA[index]), 'rb') as f_A:
            imgA = Image.open(f_A)
            imgA.load()
        imgA = self.transform(imgA)
        if self.mode == "train":
            with open(os.path.join(self.pathB, self.imgsB[index]), 'rb') as f_B:
                imgB = Image.open(f_B)
                imgB.load()
            imgB = self.transform(imgB)
        else:
            imgB = torch.zeros_like(imgA)
        if self.mode == "train":
            i, j, h, w = transforms.RandomCrop.get_params(
                imgA, output_size=(self.crop_size, self.crop_size))
            imgA = TF.crop(imgA, i, j, h, w)
            imgB = TF.crop(imgB, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                imgA = TF.hflip(imgA)
                imgB = TF.hflip(imgB)

            # Random vertical flipping
            if random.random() > 0.5:
                imgA = TF.vflip(imgA)
                imgB = TF.vflip(imgB)

        return imgA, imgB

    def __len__(self):
        return len(self.imgsA)
