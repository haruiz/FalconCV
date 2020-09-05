import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from falconcv.models.detectron import ModelZoo


class DtConfig(object):
    def __init__(self, model: str):
        self._cfg = None
        self._train_class_names = None
        self._load(model)

    @property
    def cfg(self):
        return self._cfg

    def _load(self, model: str):
        # get model info
        model_info = ModelZoo.get_model_info(model)
        # create configuration file
        self._cfg = get_cfg()
        self._cfg.merge_from_file(model_zoo.get_config_file(model_info.get_config_path()))
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_info.get_config_path())
        self._train_class_names = MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

    def update_threshold(self, threshold: float):
        self._cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self._cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold

    def update_top_k(self, top_k: int):
        self._cfg.TEST.DETECTIONS_PER_IMAGE = top_k

    def update_for_train(self, epochs: int, lr: float, bs: int, train_ds_name: str, test_ds_name: str,
                         num_classes: int, output_folder: str):
        self._cfg.SOLVER.MAX_ITER = epochs
        self._cfg.DATASETS.TRAIN = (train_ds_name,)
        self._train_class_names = MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]).get("thing_classes", None)
        self._cfg.DATASETS.TEST = ()
        if test_ds_name is not None:
            self._cfg.DATASETS.TEST = (test_ds_name,)
        self._cfg.DATALOADER.NUM_WORKERS = 0
        self._cfg.SOLVER.IMS_PER_BATCH = 1
        self._cfg.SOLVER.BASE_LR = lr
        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = bs
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self._cfg.OUTPUT_DIR = output_folder

    def get_class_label(self, class_index: np.int64):
        if self._train_class_names is None:
            return "N/A"
        return self._train_class_names[class_index]
