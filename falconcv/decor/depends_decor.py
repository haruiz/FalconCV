import inspect
import logging
import sys

# from pprint import  pprint
from functools import wraps

log = logging.getLogger("rich")


def depends(*decor_args, **decor_kargs):
    optional = decor_kargs.get("optional", False)

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                not_found_modules = [
                    pck for pck in decor_args if pck not in sys.modules
                ]
                if not_found_modules or optional:
                    msg = "Module(s) not found: {}".format(not_found_modules)
                    raise ModuleNotFoundError(msg)
                return func(*args, **kwargs)
            except Exception as ex:
                logging.exception(f"Error in {func.__qualname__}: {ex}")
                raise

        return wrapper

    return decorate


def requires(*args):
    def decor(cls):
        method_list = inspect.getmembers(
            cls, predicate=lambda x: inspect.ismethod(x) or inspect.isfunction(x)
        )
        for name, fn in method_list:
            setattr(cls, name, depends(*args)(fn))
        return cls

    return decor
