import logging
import os

from .trainable import TrainableModel
from .trained import SaveModel, FreezeModel

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class APIModelFactory:
    @staticmethod
    def create(model=None, labels_map=None, config=None):
        if config is None:
            if os.path.isfile(model):
                return FreezeModel(model, labels_map)
            else:
                return SaveModel(model, labels_map)
        else:
            return TrainableModel(config)  # model ready for training
