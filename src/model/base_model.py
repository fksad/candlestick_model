# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 00:24
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
