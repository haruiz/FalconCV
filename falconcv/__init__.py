import os
import warnings
import logging
import tensorflow as tf
import falconcv.models

from colorlog import ColoredFormatter
from .cons import *

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

tf.get_logger().setLevel(logging.ERROR)

name = "falconcv"
__version__ = "1.0.20"

# TFODAPI Environment Vars
os.environ["TF_OBJECT_DETECTION_MODEL_ZOO_URI"] = \
    r"https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md"
os.environ["TF_OBJECT_DETECTION_MODEL_CONFIG_URI"] = \
    r"https://api.github.com/repos/tensorflow/models/contents/research/object_detection/samples/configs"
