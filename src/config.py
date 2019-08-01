from .utils import load_yaml, save_yaml


class Config(object):

    @classmethod
    def from_yaml(cls, f):
        return cls(load_yaml(f))

    def __init__(self, data=None):
        self._data = data or {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def save(self, f: str):
        save_yaml(self._data, f)

    def keys(self):
        return self._data.keys()
