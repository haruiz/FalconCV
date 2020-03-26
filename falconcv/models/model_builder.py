import logging
from falconcv.cons import *
from falconcv.models.api_model import ApiModel

logger=logging.getLogger(__name__)


class ModelBuilder:
    @classmethod
    def build(cls,model=None, labels_map=None, config=None, backend=TENSORFLOW):
        try:
            # create model base on the parameters
            if backend == TENSORFLOW:
                from falconcv.models.tf import APIModelFactory
                return APIModelFactory.create(model,labels_map, config)
            elif backend == PYTORCH:
                raise NotImplementedError("Not implemented yet")
            else:
                raise NotImplementedError("Invalid backend parameter")
        except Exception as ex:
            logger.error("Error creating the model: {}".format(ex))
