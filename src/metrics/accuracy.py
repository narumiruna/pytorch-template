from __future__ import division

import torch


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            self.correct += output.argmax(dim=1).eq(target).sum().item()
            self.count += target.size(0)

    @property
    def value(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.value * 100)

    def __lt__(self, other):
        if isinstance(other, Accuracy):
            other = other.value
        return self.value < other

    def __gt__(self, other):
        if isinstance(other, Accuracy):
            other = other.value
        return self.value > other
