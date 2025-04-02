import json
from pathlib import Path

import numpy as np
import torch
import yaml


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_yaml(f):
    with open(f) as fp:
        return yaml.safe_load(fp)


def save_yaml(data, f, **kwargs):
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fp:
        yaml.safe_dump(data, fp, **kwargs)


def load_json(f):
    data = None
    with open(f) as fp:
        data = json.load(fp)
    return data


def save_json(data, f, **kwargs):
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fp:
        json.dump(data, fp, **kwargs)
