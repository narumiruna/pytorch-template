import time

import torch

from ..metrics import Average
from .log import get_logger

LOGGER = get_logger(__name__)


def sync_perf_counter():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter()


def timeit(func):
    average = Average()

    def timed(*args, **kwargs):
        start = sync_perf_counter()
        output = func(*args, **kwargs)
        t = sync_perf_counter() - start

        average.update(t)

        LOGGER.info('%s took %.6f seconds, average: %s seconds.', func.__qualname__, t, average)
        return output

    return timed
