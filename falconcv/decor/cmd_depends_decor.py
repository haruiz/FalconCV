import logging
import sys

# from pprint import  pprint
from functools import wraps
from shutil import which

log = logging.getLogger("rich")


def cmd_depends(*decor_args):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                cmd_tools_require = {
                    cmd_tool: which(cmd_tool) for cmd_tool in decor_args
                }
                for name, path in cmd_tools_require.items():
                    if path:
                        log.info(f"tool {name} found at {path}")
                    else:
                        log.info(f"tool {name} not found")

                if all(list(cmd_tools_require.values())):
                    return func(*args, **kwargs)
                msg = f"cmd {name} not found, please install it first"
                raise Exception(msg)
            except Exception as ex:
                log.error(ex)
                raise

        return wrapper

    return decorate
