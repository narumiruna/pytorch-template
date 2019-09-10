import json
import os


def load_json(f):
    data = None
    with open(f, 'r') as fp:
        data = json.load(fp)
    return data


def save_json(data, f, **kwargs):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, 'w') as fp:
        json.dump(data, fp, **kwargs)


def get_factory(obj):

    class Factory(object):

        @staticmethod
        def create(*args, **kwargs):
            name = kwargs.pop('name')
            return getattr(obj, name)(*args, **kwargs)

    return Factory
