import numpy as np
import logging
import abc
from pathlib import Path

from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

from falconcv.models import ApiModel
from falconcv.decor import typeassert
from falconcv.util.img_util import ImageUtil
from falconcv.util import BoundingBox
from .config import DtConfig
from .pascal_voc_ds import DtPascalVOCDataset

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

    def output(self, input_image: np.ndarray):
        if self._dt_config and self._predictor:
            predictions = self._predictor(input_image)
            return predictions
        return None


class DtSaveModel(DtTrainedModel):
    @typeassert(model=str, config=dict)
    def __init__(self, model: str, config: dict):
        logger.info("[INFO] Detectron2 predictions...")
        super(DtSaveModel, self).__init__(model)
        self._config = config
        # validate params
        self._check_params(config)
        # init props
        self._model_path = None
        self._model_zoo_config = None
        self._dataset_folder = None
        self._dataset_xml_folder = None
        self._ds_name = None
        self._labels_map = None
        # unwrap config in props
        self._unwrap_config(config)
        # init props
        self._classes_list = list(self._labels_map.keys())
        self._num_classes = len(self._classes_list)
        self._dataset = DtPascalVOCDataset(self._classes_list)
        self._dt_config = None

    def __enter__(self):
        logger.info("[INFO] Loading model...")
        try:
            self._dataset.register(self._ds_name, self._dataset_folder, self._dataset_xml_folder, "test")
            self._dt_config = DtConfig(self._model, self._model_zoo_config)
            self._dt_config.update_for_inference(self._ds_name, self._num_classes)
        except Exception as ex:
            raise Exception(f"[ERROR] Error loading the model: {ex}") from ex
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(DtSaveModel, self).__exit__(exc_type, exc_val, exc_tb)

    def _check_params(self, config: dict):
        # check model
        assert "model" in config and isinstance(config["model"], str), \
            "`model` parameter is required and must be an string"

        # check model zoo config
        assert "model_zoo_config" in config and isinstance(config["model_zoo_config"], str), \
            "`model zoo config` parameter is required and must be an string"

        # check dataset folder
        assert "dataset_folder" in config and isinstance(config["dataset_folder"], str), \
            "`dataset folder` parameter is required and must be an string"

        assert "labels_map" in config, "`labels map` parameter is required and must be a dictionary or a file"

    def _unwrap_config(self, config: dict):
        # reading model path
        self._model_path = config["model"]
        self._model_zoo_config = config["model_zoo_config"]

        # reading dataset images folder
        self._dataset_folder = Path(config["dataset_folder"])
        assert self._dataset_folder.exists(), "dataset folder not found"

        self._dataset_xml_folder = Path(config.get("dataset_xml_folder", self._dataset_folder))
        self._ds_name = "ds_test"

        # reading labels maps
        labels_map = config.get("labels_map", None)
        if labels_map:
            if isinstance(labels_map, dict):
                self._labels_map = labels_map
            else:
                raise Exception("`labels map` parameter must be a dictionary or a file")

    def output(self, input_image: np.ndarray):
        if self._dt_config and self._predictor:
            predictions = self._predictor(input_image)
            return predictions
        return None
