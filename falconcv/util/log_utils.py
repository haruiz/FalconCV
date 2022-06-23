import inspect
import sys
import typing


class LogUtils:
    @staticmethod
    def get_class(f: typing.Callable) -> typing.Union[typing.Type, None]:
        module_name = f.__module__
        if module_name in sys.modules:
            module = sys.modules[module_name]
            module_vars = vars(module)
            cls_name = f.__qualname__.split(".")[0]
            if cls_name in module_vars:
                return module_vars[cls_name]
        return None
