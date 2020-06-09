import logging
import os

from .trainable import TfTrainableModel
from .trained import TfSaveModel, TfFreezeModel

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class APIModelFactory:
    @staticmethod
    def create(model=None, labels_map=None, config=None):
        if config is None:
            if os.path.isfile(model):
                return TfFreezeModel(model, labels_map)
            else:
                return TfSaveModel(model, labels_map)
        else:
            return TfTrainableModel(config)  # model ready for training
