import numpy as np
import logging
import abc

from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

from falconcv.models import ApiModel
from falconcv.decor import typeassert
from falconcv.util.img_util import ImageUtil
from falconcv.util import BoundingBox
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

    @typeassert(input_image=[str, np.ndarray], size=tuple, threshold=float, top_k=int)
    def __call__(self, input_image, size=None, threshold=0.5, top_k=10):
        logger.info("[INFO] Pre-processing image...")
        img_arr, img_width, img_height, scale_factor = ImageUtil.process_input_image(input_image, size)

        logger.info("[INFO] Making predictions...")
        self._dt_config.update_threshold(threshold)
        self._dt_config.update_top_k(top_k)
        if self._predictor is None:
            self._predictor = DefaultPredictor(self._dt_config.cfg)
        output_dict = self.output(img_arr)

        logger.info("[INFO] Building annotations...")
        output_dict = output_dict["instances"].to("cpu")
        boxes = output_dict.pred_boxes.tensor.numpy() if output_dict.has("pred_boxes") else None
        scores = output_dict.scores.numpy() if output_dict.has("scores") else None
        classes = output_dict.pred_classes.numpy().astype(np.int64) if output_dict.has("pred_classes") else None
        num_detections = len(output_dict)
        masks = [None for _ in range(num_detections)]
        predictions = []

        for box, mask, score, label in zip(boxes, masks, scores, classes):
            label = self._dt_config.get_class_label(label)
            start_x, start_y, end_x, end_y = box
            predictions.append(BoundingBox(
                start_x,
                start_y,
                end_x,
                end_y,
                label,
                round(float(score), 2),
                scale_factor,
                mask
            ))
        return img_arr, predictions


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
