from .trained import DtFreezeModel
from .trainable import DtTrainableModel


class APIModelFactory:
    @staticmethod
    def create(model=None, config=None):
        if config is None:
            return DtFreezeModel(model)
        else:
            return DtTrainableModel(config)
