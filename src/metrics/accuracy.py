from __future__ import division


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, pred, label):
        correct = pred.eq(label.data).sum().item()

        self.correct += correct
        self.count += pred.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count
