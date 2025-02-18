# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 00:50
from pydantic import BaseModel


class Metric(BaseModel):
    loss: float = 0
    accuracy: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0

    def __str__(self):
        return f"loss: {self.loss:.4f}, accuracy: {self.accuracy:.4f}, precision: {self.precision:.4f}, recall: {self.recall:.4f}, f1_score: {self.f1_score:.4f}"