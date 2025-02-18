# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 01:47
import unittest

from src.metric import Metric, MetricSequence


class MetricTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_something(self):
        metric_1 = Metric(loss=1)
        metric_2 = Metric(recall=2)
        metric_sequence = MetricSequence([metric_1, metric_2])
        squeezed_metric = metric_sequence.squeeze()
        self.assertEqual(squeezed_metric.loss, .5)
        self.assertEqual(squeezed_metric.recall, 1)
        self.assertEqual(squeezed_metric.accuracy, 0)
        self.assertEqual(squeezed_metric.precision, 0)
        self.assertEqual(squeezed_metric.dict(), metric_sequence.gen_avg_value())

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
