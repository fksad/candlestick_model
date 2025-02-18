# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 00:01
from typing import List

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch import nn
from torch.types import Device
from torchmetrics import Accuracy, Precision, Recall, F1Score

from src.metric.metric import Metric


def plot_loss(train_loss_list: List[float], val_loss_list: List[float]) -> None:
    # 绘制损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def gen_metric(outputs: Tensor, label: Tensor, device: Device=torch.device('cpu'), loss: nn.modules.Module=None) -> Metric:
    accuracy = Accuracy(task='binary').to(device)
    precision = Precision(task='binary').to(device)
    recall = Recall(task='binary').to(device)
    f1 = F1Score(task='binary').to(device)
    acc = accuracy(outputs, label)
    prec = precision(outputs, label)
    rec = recall(outputs, label)
    f1 = f1(outputs, label)
    return Metric(loss=loss.item() if loss is not None else 0,
                  accuracy=acc.item(), precision=prec.item(), recall=rec.item(), f1_score=f1.item())
