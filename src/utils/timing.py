from time import time

from .log import get_logger

LOGGER = get_logger(__name__)


def timeit(func):
    def timed(*args, **kwargs):
        start = time()
        output = func(*args, **kwargs)
        LOGGER.info('%s took %.6f seconds.', func.__qualname__, time() - start)
        return output

    return timed
