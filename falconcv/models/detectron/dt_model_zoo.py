import os
import requests
import logging

from urllib.parse import urlparse
from bs4 import BeautifulSoup

from ...util.config_util import ConfigUtil
from ..model_info import ModelInfo

logger = logging.getLogger(__name__)


class DetectronModelZoo:
    @staticmethod
    def get_available_models(task=None) -> []:
        try:
            models = []
            zoo_url = ConfigUtil().get_detectron_model_zoo_url()
            result = requests.get(zoo_url)
            if result.status_code == 200:
                soup = BeautifulSoup(result.content, 'lxml')
                for tbl in soup.find_all("table"):
                    for td in tbl.select("tbody tr"):
                        model_info = ModelInfo()
                        for a in td.find_all("a", href=True):
                            href = a["href"]
                            path = urlparse(href).path
                            ext = os.path.splitext(path)[1]
                            if ext == ".yaml":
                                model_info.name = a.get_text()
                                model_info.config_url = "github.com" + href
                            elif ext == ".pkl":
                                if "COCO-Detection" in href:
                                    model_info.task = "detection"
                                elif "COCO-InstanceSegmentation" in href:
                                    model_info.task = "instance"
                                elif "COCO-Keypoints" in href:
                                    model_info.task = "keypoints"
                                elif "COCO-PanopticSegmentation" in href:
                                    model_info.task = "panoptic"
                                elif "LVIS-InstanceSegmentation" in href:
                                    model_info.task = "lvis"
                                model_info.url = href

                        if model_info.task is not None:
                            models.append(model_info)

            if task:
                assert task in ["detection", "instance", "keypoints", "panoptic", "lvis"], "Invalid task param"
                models = [model for model in models if model.task == task]

            return models
        except Exception as ex:
            logger.error("Error listing the models : {}".format(ex))

    @classmethod
    def print_available_models(cls, task="detection"):
        print("*** Detectron2 Model Zoo ***")
        models = cls.get_available_models(task)
        if models is not None:
            _ = [print(model) for model in models]
