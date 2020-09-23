from .trained import DtFreezeModel, DtSaveModel
from .trainable import DtTrainableModel


class APIModelFactory:
    @staticmethod
    def create(model=None, config=None):
        if model is not None and config is not None:
            return DtSaveModel(model, config)
        elif config is None:
            return DtFreezeModel(model)
        else:
            return DtTrainableModel(config)
