import os
import markdown
import requests
import logging

from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from falconcv.util import FileUtil, LibUtil
from pick import pick

from ...decor import typeassert

logger = logging.getLogger(__name__)


class TFModelZoo:
    @classmethod
    def pick_od_model(cls):
        # pip install windows-curses
        return pick(list(cls.available_models().keys()), "pick the model")[0]

    @classmethod
    def pick_od_pipeline(cls):
        # pip install windows-curses
        return pick(list(cls.available_pipelines().keys()), "pick the pipeline")[0]

    @staticmethod
    def available_models(arch=None) -> []:
        try:
            models = {}
            r = requests.get(os.environ["TF_OBJECT_DETECTION_MODEL_ZOO_URI"])
            if r.status_code == 200:
                md = markdown.Markdown()
                html = md.convert(r.text)
                soup = BeautifulSoup(html, "lxml")
                for a in soup.find_all('a', href=True):
                    model_url = a['href']
                    model_name = a.get_text()
                    path = urlparse(model_url).path
                    ext = os.path.splitext(path)[1]
                    if ext == ".gz":
                        models[model_name] = model_url
            if arch:
                assert arch in ["ssd", "faster", "mask"], "Invalid arch param"
                models = {k: v for k, v in models.items() if k.startswith(arch)}

            return models
        except Exception as e:
            logger.error("Error listing the models : {}".format(e))

    @classmethod
    def available_pipelines(cls):
        try:
            uri = os.environ["TF_OBJECT_DETECTION_MODEL_CONFIG_URI"]
            response = requests.get(uri)
            config_files = {}
            if response.status_code == 200:
                for f in response.json():
                    name = f["name"]
                    url = f["html_url"] \
                        .replace("blob", "raw")
                    filename, ext = os.path.splitext(name)
                    if ext == ".config":
                        config_files[filename] = url

            return config_files
        except Exception as e:
            logger.error("Error listing the pipelines : {}".format(e))

    @classmethod
    @typeassert(model_name=str)
    def download_pipeline(cls, model_name: str) -> str:
        available_pipelines = cls.available_pipelines()
        assert model_name in available_pipelines, \
            "there is not a pipeline available for the model {}".format(model_name)
        pipeline_uri = available_pipelines[model_name]
        filename = Path(urlparse(pipeline_uri).path).name
        pipeline_model_path = LibUtil.pipelines_home(subfolder="tf").joinpath(filename)
        if not pipeline_model_path.exists():
            pipeline_model_path = FileUtil.download_file(pipeline_uri, pipeline_model_path)

        return str(pipeline_model_path)

    @classmethod
    @typeassert(model_name=str)
    def download_model(cls, model_name: str) -> str:
        available_models = cls.available_models()  # get the lis
        assert model_name in available_models, "Invalid model name {}".format(model_name)
        checkpoint_model_path = LibUtil.models_home(subfolder="tf").joinpath(model_name)
        if not checkpoint_model_path.exists():
            model_uri = available_models[model_name]
            FileUtil.download_and_unzip_file(model_uri, checkpoint_model_path)

        return str(checkpoint_model_path)

    @classmethod
    def print_available_models(cls, arch=None):
        print("*** TensorFlow Detection Model Zoo ***")
        models = cls.available_models(arch)
        if models is not None:
            _ = {model: print(f"  {model}") for model in models.keys()}
