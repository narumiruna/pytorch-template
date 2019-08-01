from .utils import load_json, load_yaml, save_json, save_yaml


class Config(object):

    @classmethod
    def from_yaml(cls, f):
        return cls(load_yaml(f))

    @classmethod
    def from_args(cls, args):
        return cls(load_json(args.config_file))

    def __init__(self, data=None):
        self._data = data or {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def save(self, f: str):
        save_json(self._data, f, indent=4)

    def keys(self):
        return self._data.keys()
