import json
from pathlib import Path

import numpy as np
import torch
import yaml


def manual_seed(seed: int = 0) -> None:
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_yaml(f: str | Path) -> dict:
    with open(f) as fp:
        return yaml.safe_load(fp)


def save_yaml(data: dict, f: str | Path, **kwargs) -> None:
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fp:
        yaml.safe_dump(data, fp, **kwargs)


def load_json(f: str | Path) -> dict:
    with open(f) as fp:
        return json.load(fp)


def save_json(data: dict, f: str | Path, **kwargs) -> None:
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fp:
        json.dump(data, fp, **kwargs)
