from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode

from falconcv.models.detectron import ModelZoo


class DtConfig(object):
    def __init__(self, model: str):
        self._cfg = None
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

    def update_threshold(self, threshold: float):
        self._cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self._cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold

    def update_top_k(self, top_k: int):
        self._cfg.TEST.DETECTIONS_PER_IMAGE = top_k
