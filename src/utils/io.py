import json
import os


def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


def save_json(obj, f, **kwargs):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, 'w') as fp:
        json.dump(obj, fp, **kwargs)
