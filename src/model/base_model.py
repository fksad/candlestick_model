# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 00:24
from abc import ABC, abstractmethod

from src.metric.metric import Metric


class BaseModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Metric:
        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load(self, *args, **kwargs):
        raise NotImplementedError
