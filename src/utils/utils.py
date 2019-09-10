import json
import os

import numpy as np
import torch
import yaml


def load_json(f):
    data = None
    with open(f, 'r') as fp:
        data = json.load(fp)
    return data


def save_json(data, f, **kwargs):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, 'w') as fp:
        json.dump(data, fp, **kwargs)


def load_yaml(f):
    data = None
    with open(f, 'r') as fp:
        data = yaml.safe_load(fp)
    return data


def save_yaml(data, f, **kwargs):
    with open(f, 'w') as fp:
        yaml.safe_dump(data, fp, **kwargs)


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)
