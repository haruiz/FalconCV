import inspect
import logging
import typing
from functools import wraps
from inspect import signature
from typing_extensions import get_origin, get_args

log = logging.getLogger("rich")


def typeassert(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_sig = signature(func)
        func_args_signature = func_sig.parameters
        # arg values from calling function
        func_args = func_sig.bind(*args, **kwargs).arguments
        # Check type assertions for all arguments
        for arg_name, arg_value in func_args.items():
            arg_signature = func_args_signature[arg_name]
            arg_annotation = arg_signature.annotation
            if arg_annotation is inspect.Parameter.empty or arg_value is None:
                continue
            if (
                isinstance(arg_annotation, list)
                or get_origin(arg_annotation) == typing.Union
            ):
                valid_types = (
                    arg_annotation
                    if isinstance(arg_annotation, list)
                    else list(get_args(arg_annotation))
                )
                type_checking = [isinstance(arg_value, t) for t in valid_types]
                if not any(type_checking):
                    supported_types = ",".join(list(map(str, valid_types)))
                    log.error(
                        f"The argument `{arg_name}` of the function `{func.__qualname__}` must be one of the "
                        f"following types: {supported_types}"
                    )
                    raise TypeError
            else:
                if not isinstance(arg_value, arg_annotation):
                    log.error(
                        f"argument `{arg_name}` must be `{arg_annotation}` in the function `{func.__qualname__}`"
                    )
                    raise TypeError
        return func(*args, **kwargs)

    return wrapper
