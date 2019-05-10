import unittest

import torch

from src.models import LeNet


class TestModels(unittest.TestCase):

    def test_lenet(self):
        m = LeNet()
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            y = m(x)

        self.assertListEqual(list(y.size()), [1, 10])


if __name__ == '__main__':
    unittest.main()
