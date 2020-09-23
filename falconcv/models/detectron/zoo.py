import os
import requests
import logging

from urllib.parse import urlparse
from bs4 import BeautifulSoup

from falconcv.util.config_util import ConfigUtil
from falconcv.models.model_info import ModelInfo

logger = logging.getLogger(__name__)


class ModelZoo:
    _models = None

    @classmethod
    def available_models(cls, task=None) -> []:
        try:
            models = []
            zoo_url = ConfigUtil.get_detectron_model_zoo_url()
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

            cls._models = models

            if task:
                assert task in ["detection", "instance", "keypoints", "panoptic", "lvis"], "Invalid task param"
                models = [model for model in models if model.task == task]

            return models
        except Exception as ex:
            logger.error(f"[ERROR] Error listing the models: {ex}")

    @classmethod
    def get_model_info(cls, model: str) -> ModelInfo:
        if cls._models is None:
            cls.available_models()

        model_info = [model_info for model_info in cls._models if model_info.name == model]
        if model_info is None:
            logger.error(f"[ERROR] Model '{model}' not found in Detectron2 model zoo")
            raise ValueError(f"Model '{model}' not found in Detectron2 model zoo")

        return next(iter(model_info), None)

    @classmethod
    def print_available_models(cls, task="detection"):
        print("*** Detectron2 Model Zoo ***")
        print(*cls.available_models(task), sep="\n")
