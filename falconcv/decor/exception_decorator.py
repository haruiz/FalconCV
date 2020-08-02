import inspect
import logging
import sys
# from pprint import  pprint
import traceback
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def get_class(f):
    return vars(sys.modules[f.__module__])[f.__qualname__.split('.')[0]]


def exception(function):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            # exc_type, exc_value, exc_tb = sys.exc_info()
            # pprint(traceback.format_exception(exc_type, exc_value, exc_tb))
            # log the exception
            # pprint(traceback.extract_stack())
            # print(traceback.format_exception_only(type(ex), ex))
            # re-raise the exception
            # raise
            exc_type, exc_value, exc_traceback = sys.exc_info()
            exc_message = traceback.format_exception_only(exc_type, exc_value)[0]
            cls = get_class(function)
            if cls:
                logger.error(
                    " Calling the function `{}.{}` -> {} ".format(cls.__name__, function.__name__, exc_message))
            else:
                logger.error(" Calling the function `{}` -> {} ".format(function.__name__, exc_message))
            raise

    return wrapper
