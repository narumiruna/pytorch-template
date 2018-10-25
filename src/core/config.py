import os

from ..utils import load_json, save_json


class Config(object):

    def __init__(self):
        self.data = None
        self.basename = None

    def load_config(self, f: str):
        self.data = load_json(f)
        self.basename = os.path.basename(f)

    def save_config(self, f: str):
        save_json(self.data, f, indent=4)
