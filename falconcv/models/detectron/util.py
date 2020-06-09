from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode

from falconcv.models.detectron import ModelZoo


class Utilities:
    @staticmethod
    def load_config(model: str) -> CfgNode:
        # get model info
        model_info = ModelZoo.get_model_info(model)
        # create configuration file
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_info.get_config_path()))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_info.get_config_path())
        return cfg

    @staticmethod
    def update_config(cfg: CfgNode, threshold: float, top_k: int) -> CfgNode:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
        cfg.TEST.DETECTIONS_PER_IMAGE = top_k
        return cfg
