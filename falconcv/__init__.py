import logging
import os
import typing
from pathlib import Path

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False)],
)

__params__ = dict()


def set_home(new_path: typing.Union[Path, str]):
    __params__["HOME"] = Path(new_path)  # change default home folder


def home():
    if "HOME" in __params__:
        home_folder = __params__["HOME"]
    else:
        home_folder = Path().home().joinpath(f".{__name__}")
    home_folder.mkdir(exist_ok=True, parents=True)
    return home_folder


# constants
__version__ = "1.0.0"

os.environ[
    "TF_OBJECT_DETECTION_MODELS_ZOO_URI"
] = r"https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md"
os.environ[
    "TF_OBJECT_DETECTION_MODELS_CONFIG_URI"
] = r"https://api.github.com/repos/tensorflow/models/contents/research/object_detection/configs/tf2"
