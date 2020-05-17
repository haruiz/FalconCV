import inspect
from falconcv.cons import *
from falconcv.decor import exception


class ModelBuilder:
    @classmethod
    @exception
    def build(cls, model=None, labels_map=None, config=None, backend=TENSORFLOW):
        # create model base on the parameters
        if backend == TENSORFLOW:
            from falconcv.models.tf import APIModelFactory
            return APIModelFactory.create(model, labels_map, config)
        elif backend == DETECTRON:
            from falconcv.models.detectron import DetectronModelFactory
            return DetectronModelFactory.create(model, labels_map, config)
        else:
            raise NotImplementedError("Invalid backend parameter")
