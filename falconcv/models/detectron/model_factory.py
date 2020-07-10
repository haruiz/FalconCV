import os

from falconcv.models.detectron.trained import DtFreezeModel


class APIModelFactory:
    @staticmethod
    def create(model=None, config=None):
        if config is None:
            if not os.path.isfile(model):
                return DtFreezeModel(model)
