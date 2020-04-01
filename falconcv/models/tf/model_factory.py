import logging
import os
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .tf_trainable import TfTrainableModel
from .tf_trained import TfSaveModel, TfFreezeModel



class APIModelFactory:
    @staticmethod
    def create(model=None, labels_map=None, config=None):
        if config is None:
            if os.path.isfile(model):
                return TfFreezeModel(model, labels_map)
            else:
                return TfSaveModel(model, labels_map)
        else:
            return None#TfTrainableModel(config)  # model ready fro training
