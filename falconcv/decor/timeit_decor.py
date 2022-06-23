import logging
from functools import wraps
import time

log = logging.getLogger("rich")


def timeit(func):
    @wraps(func)
    def timed(*args, **kw):
        t_start = time.time()
        output = func(*args, **kw)
        t_end = time.time()
        log.info(f'"{func.__name__}" took {(t_end - t_start) * 1000:.3f} ms to execute')
        return output

    return timed
