import logging

from .trained import DtFreezeModel

logger = logging.getLogger(__name__)


class APIModelFactory:
    @staticmethod
    def create(model=None, labels_map=None, config=None):
        if model is not None and config is not None:
            return DtFreezeModel(model, labels_map, config)
