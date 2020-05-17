import os
import logging

from .dt_trained import PreTrainedModel

logger = logging.getLogger(__name__)


class DetectronModelFactory:
    @staticmethod
    def create(model=None, labels_map=None, config=None, task="detection"):
        if config is None:
            if not os.path.isfile(model):
                return PreTrainedModel(model, task)
        #     else:
        #         return TfSaveModel(model, labels_map)
        # else:
        #     return TfTrainableModel(config)  # model ready for training
