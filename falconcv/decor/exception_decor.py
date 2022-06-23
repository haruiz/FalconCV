import logging
import sys

import traceback
from functools import wraps

log = logging.getLogger("rich")


def exception(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            exc_message = traceback.format_exception_only(exc_type, exc_value)[0]
            log.exception(
                " Calling the function `{}` -> {} ".format(
                    function.__qualname__, exc_message
                )
            )
            raise

    return wrapper
