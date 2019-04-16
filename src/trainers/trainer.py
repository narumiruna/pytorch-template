import torch


class Trainer(object):

    def __init__(self, output_dir, use_cuda):
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() and
                                   use_cuda else 'cpu')

    def run(self):
        raise NotImplementedError
