from abc import ABCMeta, abstractclassmethod

import torch


class AbstractTrainer(metaclass=ABCMeta):
    @abstractclassmethod
    def fit(self):
        raise NotImplementedError

    @abstractclassmethod
    def train(self):
        raise NotImplementedError

    @abstractclassmethod
    def evaluate(self):
        raise NotImplementedError


class Trainer(AbstractTrainer):
    def __init__(self, output_dir, use_cuda):
        self.output_dir = output_dir
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    def run(self):
        raise NotImplementedError
