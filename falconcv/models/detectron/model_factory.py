import os
import logging

from falconcv.models.detectron.trained import DtFreezeModel
from falconcv.models import ModelConfig

logger = logging.getLogger(__name__)


class APIModelFactory:
    @staticmethod
    def create(model_config: ModelConfig):
        if model_config.is_pretrained_model:
            return DtFreezeModel(model_config)
