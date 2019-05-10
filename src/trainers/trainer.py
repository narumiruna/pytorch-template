from abc import ABCMeta, abstractmethod


class AbstractTrainer(metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
