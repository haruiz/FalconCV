import logging
import os
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    from .tf_trainable import TfTrainableModel
    from .tf_trained import TfSaveModel,TfFreezeModel,TfTrainedModel
except ImportError as ex:
   logging.error("Dependencies error: {}".format(ex))

logger=logging.getLogger(__name__)

class APIModelFactory:
    @staticmethod
    def create(model=None,labels_map=None,config=None):
        if config is None:
            if os.path.isfile(model):
                return TfFreezeModel(model,labels_map)
            else:
                return TfSaveModel(model,labels_map)
        else:
            return TfTrainableModel(config) # model ready fro training
