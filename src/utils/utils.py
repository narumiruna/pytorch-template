import json
import os

import yaml


def load_yaml(f):
    with open(f, 'r') as fp:
        return yaml.safe_load(fp)


def save_yaml(data, f, **kwargs):
    with open(f, 'w') as fp:
        yaml.safe_dump(data, fp, **kwargs)


def load_json(f):
    data = None
    with open(f, 'r') as fp:
        data = json.load(fp)
    return data


def save_json(data, f, **kwargs):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, 'w') as fp:
        json.dump(data, fp, **kwargs)
