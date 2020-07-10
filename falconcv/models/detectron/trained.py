import numpy as np
import logging
import abc

from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from falconcv.models import ApiModel
from falconcv.decor import typeassert
from falconcv.util.img_util import ImageUtil
from .config import DtConfig

logger = logging.getLogger(__name__)
setup_logger()


class DtTrainedModel(ApiModel):
    @abc.abstractmethod
    def __init__(self, model: str):
        self._model = model
        self._dt_config = None
        self._predictor = None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"[ERROR] Error loading the model: {exc_type}, {str(exc_val)}")

    @abc.abstractmethod
    @typeassert(input_image=np.ndarray)
    def output(self, input_image: np.ndarray):
        raise NotImplementedError()

    @typeassert(input_image=[str, np.ndarray], size=tuple)
    def predict(self, input_image, size=None, threshold=0.5, top_k=10):
        logger.info("[INFO] Pre-processing image...")
        img_arr, img_width, img_height, scale_factor = ImageUtil.process_input_image(input_image, size)

        logger.info("[INFO] Making predictions...")
        self._dt_config.update_threshold(threshold)
        self._dt_config.update_top_k(top_k)
        if self._predictor is None:
            self._predictor = DefaultPredictor(self._dt_config.cfg)
        predictions = self.output(img_arr)

        logger.info("[INFO] Making annotations...")
        vis_image = Visualizer(img_arr[:, :, ::-1], MetadataCatalog.get(
            self._dt_config.cfg.DATASETS.TRAIN[0]), scale=1.2)
        vis_image = vis_image.draw_instance_predictions(predictions["instances"].to("cpu"))
        return vis_image.get_image()


class DtFreezeModel(DtTrainedModel):
    @typeassert(model=str)
    def __init__(self, model: str):
        logger.info("[INFO] Detectron2 predictions...")
        super(DtFreezeModel, self).__init__(model)

    def __enter__(self):
        logger.info("[INFO] Loading pre-trained model...")
        super(DtFreezeModel, self).__enter__()
        self._dt_config = DtConfig(self._model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(DtFreezeModel, self).__exit__(exc_type, exc_val, exc_tb)

    def output(self, img_arr: np.ndarray):
        if self._dt_config and self._predictor:
            predictions = self._predictor(img_arr)
            return predictions
        return None
