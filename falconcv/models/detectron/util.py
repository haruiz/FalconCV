from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor

from falconcv.models.detectron import ModelZoo


class Utilities:
    @staticmethod
    def load_predictor(model: str, threshold: float, top_k: int) -> (CfgNode, DefaultPredictor):
        # get model info
        model_info = ModelZoo.get_model_info(model)

        # create configuration file
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_info.get_config_path()))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_info.get_config_path())
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
        cfg.TEST.DETECTIONS_PER_IMAGE = top_k
        cfg.freeze()

        # create predictor
        predictor = DefaultPredictor(cfg)

        return cfg, predictor
