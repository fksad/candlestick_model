# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 00:24
import os
import time

import torch
from torch import nn

from src.metric import MetricSequence
from src.model.base_model import BaseModel
from src.utils import gen_metric


class StockModel(BaseModel):
    def __init__(self, net: nn.Module, device: torch.device, loss_fn: nn.BCELoss):
        self._device = device
        self._net = net.to(device)
        self._loss_fn = loss_fn.to(device)

    @property
    def parameters(self):
        return self._net.parameters()

    def train(self, optimizer: torch.optim.Optimizer, data_loader: torch.utils.data.DataLoader):
        start_time = time.time()
        self._net.train()
        metric_list = MetricSequence()
        for train_images, train_labels in data_loader:
            self._net.zero_grad()
            inputs = train_images.to(self._device)
            labels = train_labels.to(self._device)
            outputs = self._net(inputs)
            outputs = outputs.view(outputs.shape[0])
            labels = labels.float()
            loss = self._loss_fn(outputs, labels)
            metric = gen_metric(outputs, labels, self._device, loss)
            metric_list.add_metric(metric)
            loss.backward()
            optimizer.step()
        train_metric = metric_list.squeeze()
        print(f'train cost time: {time.time() - start_time}')
        return train_metric

    def predict(self):
        pass

    def evaluate(self, data_loader: torch.utils.data.DataLoader):
        start_time = time.time()
        self._net.eval()
        metric_list = MetricSequence()
        with torch.no_grad():
            for val_images, val_labels in data_loader:
                inputs = val_images.to(self._device)
                labels = val_labels.to(self._device)
                outputs = self._net(inputs)
                outputs = outputs.view(outputs.shape[0])
                labels = labels.float()
                loss = self._loss_fn(outputs, labels)
                metric = gen_metric(outputs, labels, self._device, loss)
                metric_list.add_metric(metric)
        val_metric = metric_list.squeeze()
        print(f'val cost time: {time.time() - start_time}')
        return val_metric

    def save(self, path: str='.', model_name: str='model'):
        model_path = os.path.join(path, f'{model_name}.pth')
        torch.save(self._net.state_dict(), model_path)

    def load(self, path: str='.', model_name: str='model'):
        model_path = os.path.join(path, f'{model_name}.pth')
        self._net.load_state_dict(torch.load(model_path))
