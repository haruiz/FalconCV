import numpy as np
import logging
import abc

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from falconcv.models import ApiModel
from falconcv.decor import typeassert
from falconcv.util.img_util import ImageUtil
from .util import Utilities

logger = logging.getLogger(__name__)
setup_logger()


class DtTrainedModel(ApiModel):
    @abc.abstractmethod
    def __init__(self, labels_map: str, config: dict):
        self._labels_map = labels_map
        self._config_dict = config
        self._config = None

    def __enter__(self):
        self.load_labels_map()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Error loading the model:  {}, {}".format(exc_type, str(exc_val)))

    def load_labels_map(self):
        pass

    @staticmethod
    def _process_input_image(input_image, size=None):
        img_arr, img_width, img_height, scale_factor = ImageUtil.process_input_image(input_image, size)

        return img_arr, img_width, img_height, scale_factor

    @abc.abstractmethod
    @typeassert(input_image=np.ndarray)
    def output(self, input_image: np.ndarray):
        raise NotImplementedError()

    @typeassert(input_image=[str, np.ndarray], size=tuple)
    def predict(self, input_image, size=None):
        print("[INFO] pre-processing image...")
        img_arr, img_width, img_height, scale_factor = self._process_input_image(input_image, size)

        print("[INFO] making prediction...")
        predictions = self.output(img_arr)

        print("[INFO] making annotations...")
        vis_image = Visualizer(img_arr[:, :, ::-1], MetadataCatalog.get(
            self._config.DATASETS.TRAIN[0]), scale=1.2)
        vis_image = vis_image.draw_instance_predictions(predictions["instances"].to("cpu"))

        return vis_image.get_image()


class DtFreezeModel(DtTrainedModel):
    @typeassert(model=str, labels_map=str, config=dict)
    def __init__(self, model: str, labels_map: str, config: dict):
        print("[INFO] Detectron2 predictions...")
        super(DtFreezeModel, self).__init__(labels_map, config)
        self._model = model

    def __enter__(self):
        print("[INFO] loading pre-trained model...")
        super(DtFreezeModel, self).__enter__()
        self._config, self._predictor = Utilities.load_predictor(
            self._model, self._config_dict["threshold"], self._config_dict["top_k"])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(DtFreezeModel, self).__exit__(exc_type, exc_val, exc_tb)

    def output(self, img_arr: np.ndarray):
        if self._config and self._predictor:
            predictions = self._predictor(img_arr)

            return predictions

        return None
