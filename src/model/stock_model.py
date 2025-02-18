# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 00:24
import time
from typing import Callable

import torch
from torch import nn

from src.base_model import BaseModel
from src.utils import gen_metric


class StockModel(BaseModel):
    def __init__(self, net: nn.Module, device: torch.device):
        self._net = net
        self._device = device

    def train(self, epochs: int, optimizer: torch.optim.Optimizer, data_loader: torch.utils.data.DataLoader, loss_fn: Callable):
        for epoch in range(epochs):
            start_time = time.time()
            self._net.train()
            total_train_loss = 0
            train_metric_list = []
            for train_image, train_label in data_loader:
                train_image = train_image.to(self._device)
                train_label = train_label.to(self._device)
                optimizer.zero_grad()
                outputs = self._net(train_image)
                outputs = outputs.view(outputs.size(0))
                train_label = train_label.float()
                metric = gen_metric(outputs, train_label, self._device)
                train_metric_list.append(metric)
                loss = loss_fn(outputs, train_label)
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_train_loss = total_train_loss / len(data_loader)
            avg_train_metric = [sum(values) / len(values) for values in zip(*train_metric_list)]
            avg_train_acc, avg_train_prec, avg_train_recall = avg_train_metric

    def predict(self):
        pass

    def evaluate(self):
        pass