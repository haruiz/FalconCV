import os
import typing
from pathlib import Path

import numpy as np
from object_detection.utils import ops as utils_ops
from object_detection.utils.label_map_util import create_category_index_from_labelmap

from falconcv.data import (
    PascalVOCDataset,
    PascalVOCMetricsHandler,
    PascalVOCAnnotation,
    PascalVOCImage,
)
from falconcv.decor import requires, exception, typeassert
from falconcv.models.api_model import ApiModel
from falconcv.util import ImageUtils

try:

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    import tensorflow.compat.v1 as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
except ImportError:
    ...
import logging

logger = logging.getLogger("rich")


@requires("object_detection")
class TODATrainedModel(ApiModel):
    def __init__(self, save_model_path: Path, label_map_path: Path):
        self.save_model_path = save_model_path
        self.label_map_path = label_map_path
        self._model_signature = None
        self._labels_map_dict = create_category_index_from_labelmap(str(label_map_path))

    def _process_input(self, input_image, size=None):
        img_arr, scale_factor = ImageUtils.read(input_image), 1
        if size:
            img_arr, scale_factor = ImageUtils.resize(
                img_arr, width=size[0], height=[1]
            )  # resize image
        img_height, img_width = img_arr.shape[:2]
        return img_arr, img_width, img_height, scale_factor

    def _process_output(self, img_arr):
        img_expanded = np.expand_dims(img_arr, axis=0)
        img_tensor = tf.convert_to_tensor(img_expanded)
        output_dict = self._model_signature(img_tensor)
        num_detections = int(output_dict.pop("num_detections"))
        output_keys = [
            "detection_boxes",
            "detection_scores",
            "detection_classes",
            "detection_masks",
        ]
        output_dict = {
            key: value[0, :num_detections].numpy()
            for key, value in output_dict.items()
            if key in output_keys
        }
        output_dict["num_detections"] = num_detections
        return output_dict

    @property
    def labels_map(self):
        return {self._labels_map_dict[k]["name"]: k for k, v in self._labels_map_dict.items()}

    def __enter__(self):
        model = tf.compat.v2.saved_model.load(self.save_model_path)
        if "serving_default" not in model.signatures:
            raise ValueError("No serving signature found in the model")
        self._model_signature = model.signatures["serving_default"]
        return self

    @exception
    def compute_eval_metrics(
        self,
        eval_dataset: PascalVOCDataset,
        confidence_threshold=0.5,
        iou_threshold=0.5,
    ):
        labels = [cat["name"] for cat in self._labels_map_dict.values()]
        metrics_handler = PascalVOCMetricsHandler(
            self, eval_dataset, labels, confidence_threshold
        )
        confusion_matrix = metrics_handler.compute_confusion_matrix(iou_threshold)
        (
            precision,
            recall,
            f1_score,
            mAP,
        ) = metrics_handler.compute_precision_recall_f1_score_mAP(
            iou_threshold, confusion_matrix
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mAP": mAP,
            "confusion_matrix": confusion_matrix,
        }

    @typeassert
    def __call__(
        self,
        input_image: typing.Union[str, Path, np.ndarray],
        size: tuple = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        top_k: int = 10,
    ):
        img_arr, img_width, img_height, scale_factor = self._process_input(
            input_image, size
        )
        output_dict = self._process_output(img_arr)
        boxes = output_dict["detection_boxes"]
        scores = output_dict["detection_scores"]
        classes = output_dict["detection_classes"].astype(np.int64)
        num_detections = output_dict["num_detections"]
        masks = [None for _ in range(num_detections)]
        detection_masks_included = "detection_masks" in output_dict
        annotations = []
        if detection_masks_included:
            masks = output_dict["detection_masks"]
            masks = utils_ops.reframe_box_masks_to_image_masks(
                masks, boxes, img_height, img_width
            )
            masks = tf.cast(masks > mask_threshold, tf.uint8).numpy()
        detections_indexes = []
        for i, (box, mask, score, label) in enumerate(
            zip(boxes, masks, scores, classes)
        ):
            if score < threshold:
                continue
            detections_indexes.append(i)
            ymin, xmin, ymax, xmax = box
            annotations.append(
                PascalVOCAnnotation(
                    xmin=int(xmin * img_width),
                    ymin=int(ymin * img_height),
                    xmax=int(xmax * img_width),
                    ymax=int(ymax * img_height),
                    score=round(float(score), 2),
                    scale=scale_factor,
                    name=self._labels_map_dict[label]["name"],
                )
            )
        if len(annotations) > 0 and len(annotations) > top_k:
            annotations = annotations[:top_k]

        out_image = PascalVOCImage()
        out_image.image_arr = img_arr
        out_image.annotations = annotations
        if detection_masks_included and detections_indexes:
            out_mask = np.max(masks[detections_indexes], axis=0)
            out_image.mask_arr = out_mask
        return out_image

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Error loading the model:  {exc_type}, {exc_val}")
            raise exc_val
