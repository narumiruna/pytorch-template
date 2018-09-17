from __future__ import division


class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count


class MovingAverage(object):

    def __init__(self):
        self.average = None

    def update(self, value, weight=0.1):
        if self.average is None:
            self.average = value
        else:
            self.average = (1 - weight) * self.average + weight * value
