from omegaconf import OmegaConf


class Job:
    def run(self, config: OmegaConf, resume=None) -> None:
        raise NotImplementedError
