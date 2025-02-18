# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/18 01:47
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_something(self):
        self.assertEqual(True, False)
        
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
