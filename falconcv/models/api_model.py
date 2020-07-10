from abc import ABCMeta
from falconcv.decor import typeassert

import numpy as np
import logging

logger = logging.getLogger(__name__)


class ApiModel(metaclass=ABCMeta):
    def train(self, *args, **kwargs):
        return self

    def freeze(self, *args, **kwargs):
        pass
        return self

    def eval(self, *args, **kwargs):
        return self

    @typeassert(input_image=[str, np.ndarray], size=tuple, threshold=float, top_k=int)
    def predict(self, input_image, size=None, threshold=0.5, top_k=10):
        pass
