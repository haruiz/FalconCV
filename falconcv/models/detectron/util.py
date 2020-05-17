from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode

from falconcv.models.detectron import DetectronModelZoo


class Utilities:
    @staticmethod
    def get_detectron_config(model: str, task: str) -> CfgNode:
        model_info = DetectronModelZoo.get_model_info(model, task)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_info.get_config_path()))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_info.get_config_path())

        print(model_zoo.__all__)

        return cfg
