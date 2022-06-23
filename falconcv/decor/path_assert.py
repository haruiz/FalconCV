import inspect
import logging
import typing
from functools import wraps
from inspect import signature
from pathlib import Path
from typing_extensions import get_args

log = logging.getLogger("rich")


def pathassert(*decor_args, **decor_kwargs):
    def decorate(func):
        if not __debug__:
            return func
        # Map function argument names to supplied types
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_args = sig.bind(*args, **kwargs).arguments
            func_args_signature = sig.parameters

            func_args = (
                {k: v for k, v in func_args.items() if k in decor_args}
                if decor_args
                else func_args
            )

            for arg_name, arg_value in func_args.items():
                param: inspect.Parameter = func_args_signature[arg_name]
                annotation: typing.Generic = param.annotation
                if any(t in get_args(annotation) for t in (Path, str)):
                    is_a_valid_path = arg_value and Path(arg_value).exists()
                    if is_a_valid_path:
                        log.exception(f"Argument `{arg_name}` must be a valid path")
                        raise ValueError
            return func(*args, **kwargs)

        return wrapper

    return decorate
