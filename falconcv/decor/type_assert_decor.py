from inspect import signature
from functools import wraps


def typeassert(*ty_args, **ty_kargs):
    def decorate(func):
        if not __debug__:
            return func
        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kargs).arguments
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if value is None:
                        continue
                    attr_type = bound_types[name]
                    if isinstance(attr_type, list):
                        type_checking = map(lambda t: isinstance(value, t), attr_type)
                        type_checking = list(type_checking)
                        if not any(type_checking):
                            supported_types = list(map(str, attr_type))
                            raise TypeError('Error calling the function {}, argument {} must be {} '.format(func.__name__, name," or ".join(supported_types)))
                    else:
                        if not isinstance(value, attr_type):
                            raise TypeError('Error calling the function {}, argument {} must be {}'.format(func.__name__, name, attr_type))
            return func(*args, **kwargs)
        return wrapper
    return decorate


