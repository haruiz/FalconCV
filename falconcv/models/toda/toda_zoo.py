import logging
import os
import typing
from pathlib import Path
from urllib.parse import urlparse

import markdown
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table

from falconcv.decor import exception, typeassert
from falconcv.util import FileUtils, LibUtils
import validators

logger = logging.getLogger("rich")


class PretrainedModel:
    def __init__(self, name, uri, speed, mAP, outputs):
        self.name = name
        self.uri = uri
        self.speed = speed
        self.outputs = outputs
        self.mAP = mAP


class PretrainedModelPipeline:
    def __init__(self, uri):
        self.uri = uri


class TODAZoo:
    @staticmethod
    def fetch_pretrained_available_models() -> []:
        """
        Fetch the list of available model at the Tensorflow object detection model zoo
        :param arch:
        :return:
        """
        models = {}
        model_zoo_uri = os.environ["TF_OBJECT_DETECTION_MODELS_ZOO_URI"]
        r = requests.get(model_zoo_uri)
        if r.status_code == 200:
            md = markdown.Markdown()
            html = md.convert(r.text)
            soup = BeautifulSoup(html, "lxml")
            table = soup.find("table")
            for row in table.findAll("tr"):
                cols = row.findAll("td")
                if cols:
                    a = cols[0].find("a", href=True)
                    uri = a["href"]
                    name = a.get_text().strip()
                    speed = cols[1].get_text()
                    speed = float(speed) if speed.isnumeric() else 0
                    mAP = cols[2].get_text()
                    outputs = cols[3].get_text()
                    m = PretrainedModel(
                        name=name, uri=uri, speed=speed, mAP=mAP, outputs=outputs
                    )
                    models[name] = m
        return models

    @staticmethod
    def fetch_pretrained_available_pipelines() -> []:
        """
        Fetch the list of available train config files for the models at the Tensorflow object detection model zoo
        :param arch:
        :return:
        """
        uri = os.environ["TF_OBJECT_DETECTION_MODELS_CONFIG_URI"]
        response = requests.get(uri)
        config_files = {}
        if response.status_code == 200:
            for f in response.json():
                name = f["name"]
                url = f["html_url"].replace("blob", "raw")
                filename, ext = os.path.splitext(name)
                if ext == ".config":
                    config_files[filename] = PretrainedModelPipeline(url)
        return config_files

    @classmethod
    @exception
    def print_pretrained_available_models(cls):
        """
        print tensorflow object detection available models
        :return:
        """
        models = cls.fetch_pretrained_available_models()
        models = dict(sorted(models.items(), key=lambda x: x[1].mAP, reverse=True))
        if models:
            table = Table(title="*** TensorFlow Detection Model Zoo ***")
            for col_name in ["Name", "Uri", "Speed (ms)", "COCO mAP", "Outputs"]:
                table.add_column(col_name, justify="left", style="cyan", no_wrap=True)
            for m in models.values():
                table.add_row(m.name, m.uri, str(m.speed), m.mAP, m.outputs)
            console = Console()
            console.print(table)

    @classmethod
    @exception
    def print_pretrained_available_pipelines(cls):
        """
        print tensorflow object detection available config files
        :return:
        """
        pipelines = cls.fetch_pretrained_available_pipelines()
        if pipelines:
            table = Table(title="*** TensorFlow Detection Model Zoo - Configs ***")
            for col_name in ["Name", "Uri"]:
                table.add_column(col_name, justify="left", style="cyan", no_wrap=True)
            for k, v in pipelines.items():
                table.add_row(k, v.uri)
            console = Console()
            console.print(table)

    @classmethod
    @typeassert
    def download_model_pipeline(
        cls, pipeline: str, output_folder: typing.Union[str, Path] = None
    ) -> Path:
        pipeline_uri = pipeline
        if not validators.url(pipeline_uri):
            available_pipelines = cls.fetch_pretrained_available_pipelines()
            assert (
                pipeline in available_pipelines
            ), f"pipeline not found with name: {pipeline}"
            pipeline_uri = available_pipelines[pipeline].uri
        pipeline_out_folder = output_folder
        if pipeline_out_folder is None:
            pipeline_out_folder = LibUtils.pipelines_home(subfolder="toda")
        else:
            pipeline_out_folder = Path(pipeline_out_folder)
            pipeline_out_folder.mkdir(parents=True, exist_ok=True)

        return FileUtils.get_file_from_uri(
            pipeline_uri, pipeline_out_folder, show_progress=True
        )

    @classmethod
    @typeassert
    def download_model_checkpoint(
        cls, model: str, output_folder=typing.Union[str, Path]
    ) -> Path:
        model_uri = model
        if not validators.url(model_uri):
            available_models = cls.fetch_pretrained_available_models()  # get the lis
            assert model in available_models, f"Model not found with name : {model}"
            model_uri = available_models[model].uri

        checkpoint_model_folder = output_folder
        if checkpoint_model_folder is None:
            checkpoint_model_folder = LibUtils.models_home(subfolder=f"toda/{model}")
        else:
            checkpoint_model_folder = Path(checkpoint_model_folder)
            checkpoint_model_folder.mkdir(parents=True, exist_ok=True)

        return FileUtils.get_file_from_uri(
            model_uri, checkpoint_model_folder, unzip=True, show_progress=True
        )
