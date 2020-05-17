import numpy as np
import cv2
import logging

from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from .util import Utilities
from ...decor import typeassert
from ...util import ImageUtil

logger = logging.getLogger(__name__)


class PreTrainedModel:
    @typeassert(model=str, task=str)
    def __init__(self, model: str, task: str):
        self._model = model
        self._task = task

    def __enter__(self):
        self._config = Utilities.get_detectron_config(self._model, self._task)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(PreTrainedModel, self)

    @typeassert(input_image=[str, np.ndarray], size=tuple, threshold=float, top_k=int)
    def predict(self, input_image, size=None, threshold=0.5, top_k=10):
        # preprocess image
        print("[INFO] Detectron2 prediction...")
        print("[INFO] preprocessing image...")
        img_arr, img_width, img_height, scale_factor = ImageUtil.process_input_image(input_image, size)

        # configure threshold
        print("[INFO] configuring threshold...")
        self._config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        # create default predictor and predict
        print("[INFO] creating predictor...")
        predictor = DefaultPredictor(self._config)

        print("[INFO] predicting image...")
        predictions = predictor(img_arr)

        # annotate predictions
        print("[INFO] annotating predictions...")
        vis_image = Visualizer(img_arr[:, :, ::-1], MetadataCatalog.get(self._config.DATASETS.TRAIN[0]), scale=1.2)
        vis_image = vis_image.draw_instance_predictions(predictions["instances"].to("cpu"))

        return vis_image.get_image()
