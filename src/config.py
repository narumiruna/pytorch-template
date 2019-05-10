from .utils import load_json, save_json


class Config(object):

    def __init__(self, f: str):
        self._data = load_json(f)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def save(self, f: str):
        save_json(self._data, f, indent=4)

    def keys(self):
        return self._data.keys()
