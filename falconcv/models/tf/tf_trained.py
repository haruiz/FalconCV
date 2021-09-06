import abc
import logging
import os
import typing
from pathlib import Path

import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils.label_map_util import create_category_index_from_labelmap
from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

from falconcv.decor import typeassert, pathassert
from falconcv.models.api_model import ApiModel
from falconcv.util import ImageUtil, BoundingBox
from .util import Utilities
from .zoo import ModelZoo

logger = logging.getLogger(__name__)


class TfTrainedModel(ApiModel):
    @abc.abstractmethod
    def __init__(self, labels_map):
        self._labels_map = labels_map
        self._labels_map_dict = None

    def load_labels_map(self):
        assert os.path.isfile(self._labels_map), "Labels map file not found"
        self._labels_map_dict = create_category_index_from_labelmap(self._labels_map, use_display_name=True)

    def __enter__(self):
        self.load_labels_map()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Error loading the model:  {}, {}".format(exc_type, str(exc_val)))

    @staticmethod
    def _process_input_image(input_image, size=None):
        img_arr, scale_factor = ImageUtil.read(input_image), 1  # read image
        if size: img_arr, scale_factor = ImageUtil.resize(img_arr, width=size[0], height=[1])  # resize image
        img_height, img_width = img_arr.shape[:2]
        return img_arr, img_width, img_height, scale_factor

    @abc.abstractmethod
    @typeassert(input_image=np.ndarray)
    def output_dict(self, input_image: np.ndarray):
        raise NotImplementedError()

    @typeassert(input_image=[str, np.ndarray], size=tuple, threshold=float, top_k=int)
    def __call__(self, input_image, size=None, threshold=0.5, top_k=10):
        img_arr, img_width, img_height, scale_factor = self._process_input_image(input_image, size)
        output_dict = self.output_dict(img_arr)
        boxes = output_dict["detection_boxes"]
        scores = output_dict["detection_scores"]
        classes = output_dict["detection_classes"].astype(np.int64)
        num_detections = output_dict['num_detections']
        masks = [None for _ in range(num_detections)]
        predictions = []
        if 'detection_masks' in output_dict:
            # get masks
            masks = output_dict["detection_masks"]
            # adjust mask coordinates based on the images dimensions
            masks = utils_ops.reframe_box_masks_to_image_masks(masks, boxes, img_height, img_width)
            # check eager execution mode
            if tf.executing_eagerly():
                masks = tf.cast(masks > threshold, tf.uint8).numpy()
            else:
                masks_tensor = tf.cast(masks > threshold, tf.uint8)
                masks = masks_tensor.eval(session=tf.Session())

        for box, mask, score, label in zip(boxes, masks, scores, classes):
            if score >= threshold:
                if self._labels_map_dict and label in self._labels_map_dict:
                    label = self._labels_map_dict[label]["name"]
                start_y, start_x, end_y, end_x = box
                start_x = int(start_x * img_width)
                start_y = int(start_y * img_height)
                end_x = int(end_x * img_width)
                end_y = int(end_y * img_height)
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
        if len(predictions) > 0 and len(predictions) > top_k:
            predictions = predictions[:top_k]
        return img_arr, predictions


class TfFreezeModel(TfTrainedModel):
    @typeassert(freeze_model=str, labels_map=str)
    @pathassert
    def __init__(self, freeze_model: typing.Union[str, Path], labels_map: typing.Union[str, Path]):
        super(TfFreezeModel, self).__init__(labels_map)
        self._freeze_model = freeze_model
        _, ext = os.path.splitext(self._freeze_model)
        assert os.path.isfile(self._freeze_model) and ext == ".pb", "Invalid model file"
        self._graph = None
        self._session = None

    def __enter__(self):
        super(TfFreezeModel, self).__enter__()
        self._graph, self._session = Utilities.load_graph(self._freeze_model)
        return self

    @pathassert
    def evaluate(self, eval_record_path: typing.Union[str, Path], iou_threshold = 0.5, confidence_threshold = 0.5):
        detections_record_path = eval_record_path.parent.joinpath("detection.record")
        Utilities.generate_detections_record(str(eval_record_path),str(detections_record_path), self._freeze_model)
        record_iterator = tf.python_io.tf_record_iterator(path=str(detections_record_path))
        label_map = label_map_util.load_labelmap(self._labels_map)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100,
                                                                    use_display_name=True)
        data_parser = tf_example_parser.TfExampleDetectionAndGTParser()
        confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))
        image_index = 0
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            decoded_dict = data_parser.parse(example)
            image_index += 1
            if decoded_dict:
                groundtruth_boxes = decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
                groundtruth_classes = decoded_dict[standard_fields.InputDataFields.groundtruth_classes]
                detection_scores = decoded_dict[standard_fields.DetectionResultFields.detection_scores]
                detection_classes = decoded_dict[standard_fields.DetectionResultFields.detection_classes][
                    detection_scores >= confidence_threshold]
                detection_boxes = decoded_dict[standard_fields.DetectionResultFields.detection_boxes][
                    detection_scores >= confidence_threshold]
                matches = []
                if image_index % 100 == 0:
                    logger.info("Processed %d images" % (image_index))
                for i in range(len(groundtruth_boxes)):
                    for j in range(len(detection_boxes)):
                        iou = Utilities.compute_iou(groundtruth_boxes[i], detection_boxes[j])
                        if iou > iou_threshold:
                            matches.append([i, j, iou])
                matches = np.array(matches)
                if matches.shape[0] > 0:
                    # Sort list of matches by descending IOU so we can remove duplicate detections
                    # while keeping the highest IOU entry.
                    matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                    # Remove duplicate detections from the list.
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                    # our previous sort.
                    matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                    # Remove duplicate ground truths from the list.
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                for i in range(len(groundtruth_boxes)):
                    if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                        confusion_matrix[groundtruth_classes[i] - 1][
                            detection_classes[int(matches[matches[:, 0] == i, 1][0])] - 1] += 1
                    else:
                        confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
                for i in range(len(detection_boxes)):
                    if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
                        confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1
            else:
                logger.info("Skipped image %d" % (image_index))
        logger.info("Processed %d images" % (image_index))
        return Utilities.confusion_matrix_to_dataframe(confusion_matrix, categories, iou_threshold)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(TfFreezeModel, self).__exit__(exc_type, exc_val, exc_tb)
        if self._session:
            self._session.close()

    @staticmethod
    def _get_tensors_dict(graph):
        tensors_list = [
            'num_detections',
            'detection_boxes',
            'detection_scores',
            'detection_classes',
            'detection_masks'
        ]
        tensor_dict = {}
        for opt in graph.get_operations():
            if opt.name in tensors_list:
                tensor_dict[opt.name] = graph.get_tensor_by_name("{}:0".format(opt.name))
        return tensor_dict

    def output_dict(self, img_arr: np.ndarray):
        if self._graph and self._session:
            img_expanded = np.expand_dims(img_arr, axis=0)
            tensors_dict = self._get_tensors_dict(self._graph)
            image_tensor = self._graph.get_tensor_by_name('image_tensor:0')
            output_dict = self._session.run(tensors_dict, feed_dict={image_tensor: img_expanded})
            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {k: np.squeeze(v) for k, v in output_dict.items()}
            output_dict['num_detections'] = num_detections
            return output_dict
        return None


class TfSaveModel(TfTrainedModel):
    @typeassert(model=str, labels_map=str)
    @pathassert
    def __init__(self, model: typing.Union[str, Path], labels_map: typing.Union[str, Path]):
        super(TfSaveModel, self).__init__(labels_map)
        self._model = model
        self._tf_model = None

    def __enter__(self):
        super(TfSaveModel, self).__enter__()
        if not os.path.isdir(self._model):
            download_folder = ModelZoo.download_model(self._model)
            self._model = os.path.join(download_folder, "saved_model")
        self._tf_model = Utilities.load_save_model(self._model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(TfSaveModel, self).__exit__(exc_type, exc_val, exc_tb)

    def output_dict(self, img_arr: np.ndarray):
        if self._tf_model:
            img_expanded = np.expand_dims(img_arr, axis=0)
            img_tensor = tf.convert_to_tensor(img_expanded)
            output_dict = self._tf_model(img_tensor)  # just get the predictions of the model
            if tf.executing_eagerly():
                num_detections = int(output_dict.pop('num_detections'))
                output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
            else:
                with tf.Session() as sess:
                    output_dict = sess.run(output_dict)
                    num_detections = int(output_dict.pop('num_detections'))
                    output_dict = {k: np.squeeze(v) for k, v in output_dict.items()}
            output_dict['num_detections'] = num_detections
            return output_dict
        return None
