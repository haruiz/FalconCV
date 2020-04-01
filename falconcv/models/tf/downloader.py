from pathlib import Path
from urllib.parse import urlparse
from falconcv.decor import typeassert
from falconcv.util import LibUtil, FileUtil
from .zoo import ModelZoo



class Downloader:
    @classmethod
    @typeassert(model_name=str)
    def download_pipeline(cls, model_name: str) -> str:
        available_pipelines = ModelZoo.available_pipelines()
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
        available_models = ModelZoo.available_models()  # get the lis
        assert model_name in available_models, "Invalid model name {}".format(model_name)
        checkpoint_model_path = LibUtil.models_home(subfolder="tf").joinpath(model_name)
        if not checkpoint_model_path.exists():
            model_uri = available_models[model_name]
            FileUtil.download_and_unzip_file(model_uri, checkpoint_model_path)
        return str(checkpoint_model_path)

