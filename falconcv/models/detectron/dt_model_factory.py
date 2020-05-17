import logging

from .dt_trained import DetectronFreezeModel

logger = logging.getLogger(__name__)


class DetectronModelFactory:
    @staticmethod
    def create(model=None, labels_map=None, config=None):
        if model is not None and config is not None:
            return DetectronFreezeModel(model, labels_map, config)
