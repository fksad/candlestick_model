# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/15 16:27
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, label_df=None, max_len=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = sorted([path_ for path_ in os.listdir(data_dir) if path_.endswith('.png')])
        self.label_list = list(label_df['label'])
        if max_len is not None:
            self.file_list = self.file_list[:max_len]
            self.label_list = self.label_list[:max_len]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.file_list[idx])
        image = Image.open(img_path).convert('L')
        label = torch.tensor(self.label_list[idx], dtype=torch.float)
        if self.transform:
            image = self.transform(image)

        return image, label
