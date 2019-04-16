from time import time

from .log import get_logger

LOGGER = get_logger(__name__)


def timeit(func):
    from ..metrics import Average  # fix circular import
    time_average = Average()

    def timed(*args, **kwargs):
        start = time()
        output = func(*args, **kwargs)
        end = time()
        t = end - start
        time_average.update(t, number=1)
        LOGGER.info('%s took %.6f seconds, average: %s seconds',
                    func.__qualname__, t, time_average)
        return output

    return timed
