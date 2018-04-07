class Meter():
    def update(self):
        raise NotImplementedError


class AverageMeter(Meter):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = 0

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


class AccuracyMeter(Meter):
    def __init__(self):
        self.correct = 0
        self.count = 0
        self.accuracy = 0

    def update(self, correct, count):
        self.correct += correct
        self.count += count
        self.accuracy = self.correct / self.count
