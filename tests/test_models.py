
import torch

from src.models import LeNet


@torch.no_grad()
def test_lenet_forward():
    n = 2
    m = LeNet()
    x = torch.randn(n, 1, 32, 32)
    y = m(x)
    assert list(y.size()) == [n, 10]
