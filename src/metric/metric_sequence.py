# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 01:53
from typing import List, Dict

from src.metric.metric import Metric


class MetricSequence:
    _property_list = Metric.schema()['properties'].keys()

    def __init__(self, metric_list: List[Metric]=None):
        self._metric_list = metric_list or []

    @property
    def metric_list(self) -> List[Metric]:
        return self._metric_list

    def add_metric(self, metric: Metric):
        self._metric_list.append(metric)

    def gen_avg_value(self) -> Dict[str, float]:
        value_list = [sum([value for _, value in values]) / len(values) for values in zip(*self._metric_list)]
        result = {property_: value for property_, value in zip(self._property_list, value_list)}
        return result

    def squeeze(self) -> Metric:
        return Metric(**self.gen_avg_value())

    def __str__(self):
        avg_value_dict = self.gen_avg_value()
        result = ' '.join([f'{property_}: {value:.4f}' for property_, value in avg_value_dict.items()])
        return result
