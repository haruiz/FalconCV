import inspect
from inspect import signature
from functools import wraps

import typing
from pathlib import Path


def pathassert(func):
    sig = signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bounds = sig.bind(*args, **kwargs)
        params = sig.parameters
        for arg_name, arg_value in bounds.arguments.items():
            param: inspect.Parameter = params[arg_name]
            annotation: typing.Generic = param.annotation
            if annotation == typing.Union[str, Path]:
                if not Path(arg_value).exists():
                    raise TypeError(
                        'Argument `{}` must be a valid path'.format(arg_name))
        return func(*args, **kwargs)

    return wrapper
