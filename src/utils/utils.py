import json
import os


def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


def save_json(obj, f, **kwargs):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, 'w') as fp:
        json.dump(obj, fp, **kwargs)


def get_factory(obj):
    class Factory(object):
        @staticmethod
        def create(*args, **kwargs):
            name = kwargs.pop('name')
            return getattr(obj, name)(*args, **kwargs)

    return Factory
