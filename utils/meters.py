class Meter():

    def update(self):
        raise NotImplementedError


class AverageMeter(Meter):

    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


class MovingAverageMeter(Meter):

    def __init__(self):
        self.average = None

    def update(self, value, weight=0.1):
        if self.average is None:
            self.average = value
        else:
            self.average = (1 - weight) * self.average + weight * value


class AccuracyMeter(Meter):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, y):
        y_pred = output.data.argmax(dim=1)
        correct = y_pred.eq(y.data).cpu().sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count