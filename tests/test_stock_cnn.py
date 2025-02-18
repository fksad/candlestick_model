# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/17 14:02
import unittest

import pandas as pd
import torch

from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data_loader import CustomDataset
from src.net.stock_cnn import StockCNN
from src.model.stock_model import StockModel
from src.utils import plot_loss, gen_metric


class StockCNNTestCase(unittest.TestCase):
    def setUp(self):
        self._data_path = './data/20'
        self._batch_size = 64
        self._epochs = 10
        self._interval = 1
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._max_len = None
        self._model = StockModel(net=StockCNN(), device=self._device, loss_fn=nn.BCELoss())

    def test_train(self):
        optimizer = AdamW(self._model._net.parameters(), lr=0.00003, weight_decay=0.0005)
        train_data_loader, val_data_loader = self._load_train_data()
        train_loss_list = []
        val_loss_list = []
        max_val_f1 = 0
        for epoch in range(self._epochs):
            train_metric = self._model.train(optimizer, train_data_loader)
            print(f'epoch: {epoch}, train_metric: {train_metric}')
            if (epoch + 1) % self._interval == 0:
                val_metric = self._model.evaluate(val_data_loader)
                val_loss_list.append(val_metric.loss)
                train_loss_list.append(train_metric.loss)
                print(f'val metric: {val_metric}')
                if val_metric.f1_score > max_val_f1:
                    print(f'new max_val_f1_score: {val_metric.f1_score} in epoch: {epoch}, save model')
                    max_val_f1 = val_metric.f1_score
                    self._model.save(path='data', model_name='stock_cnn_best_f1.pth')
        plot_loss(train_loss_list, val_loss_list)

    def tearDown(self):
        pass

    def _load_train_data(self):
        image_transform = transforms.Compose([
            transforms.Resize((64, 60)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
        ])
        labels_df = pd.read_csv('data/label_file.csv')
        dataset = CustomDataset(data_dir=self._data_path, transform=image_transform, label_df=labels_df,
                                max_len=self._max_len)
        val_split = 0.2
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size*2, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader


if __name__ == '__main__':
    unittest.main()
