from omegaconf import DictConfig


class Job:
    def run(self, config: DictConfig, resume: str | None = None) -> None:
        raise NotImplementedError
