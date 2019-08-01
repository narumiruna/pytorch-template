import mlflow

from .utils import load_yaml, save_yaml


class Config(dict):

    @classmethod
    def from_yaml(cls, f):
        mlflow.log_artifact(f)
        return cls(load_yaml(f))

    def save(self, f: str):
        save_yaml(self, f)
