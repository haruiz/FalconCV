import os
import logging
from pathlib import Path

from detectron2.engine import DefaultTrainer

from falconcv.util import FileUtil
from falconcv.models import ApiModel
from .config import DtConfig
from .pascal_voc_ds import DtPascalVOCDataset

logger = logging.getLogger(__name__)


class DtTrainableModel(ApiModel):
    def __init__(self, config: dict):
        # validate params
        self._check_training_params(config)
        # init props
        self._model_name = None
        self._train_images_folder = None
        self._train_xml_folder = None
        self._test_images_folder = None
        self._test_xml_folder = None
        self._output_folder = None
        self._labels_map = None
        self._train_ds_name = None
        self._test_ds_name = None
        # unwrap config in props
        self._unwrap_config(config)
        # init objects
        self._epochs = None
        self._classes_list = list(self._labels_map.keys())
        self._num_classes = len(self._classes_list)
        self._dataset = DtPascalVOCDataset(self._classes_list)
        self._dt_config = None

    def __enter__(self):
        try:
            self._register_datasets()
            self._dt_config = DtConfig(self._model_name)
            return self
        except Exception as ex:
            raise Exception(f"[ERROR] Error loading the model: {ex}") from ex

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"[ERROR] Error loading the model: {exc_type}, {str(exc_val)}")
            raise exc_val

    def train(self, epochs=100, lr=0.02, bs=128, clear_folder=False):
        try:
            if clear_folder:
                FileUtil.clear_folder(self._out_folder)
            self._epochs = epochs
            # update pipeline
            self._dt_config.update_for_train(epochs, lr, bs, self._train_ds_name, self._test_ds_name,
                                             self._num_classes, str(self._out_folder))
            # train
            trainer = DefaultTrainer(self._dt_config.cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
        except Exception as ex:
            raise Exception(f"[ERROR] Error training the model : {ex}") from ex
        return super(DtTrainableModel, self).train()

    def _check_training_params(self, config: dict):
        # check model
        assert "model" in config and isinstance(config["model"], str), \
            "`model` parameter is required and must be an string"

        # check train images folder
        assert "train_images_folder" in config and isinstance(config["train_images_folder"], str), \
            "`train images folder` parameter is required and must be an string"

        # check test images folder
        if "test_images_folder" in config:
            assert isinstance(config["test_images_folder"], str), "`test images folder` parameter must be an string"

        # check output folder
        if "output_folder" in config:
            assert isinstance(config["output_folder"], str), "`output_folder` must be an string"

        assert "labels_map" in config, "`labels map` parameter is required and must be a dictionary or a file"

    def _unwrap_config(self, config: dict):
        # reading model
        self._model_name = config["model"]

        # reading train images folder
        self._train_images_folder = Path(config["train_images_folder"])
        assert self._train_images_folder.exists(), "train images folder not found"

        self._train_xml_folder = Path(config.get("train_xml_folder", self._train_images_folder))
        self._train_ds_name = "ds_train"

        # reading test images folder
        if "test_images_folder" in config:
            self._test_images_folder = Path(config["test_images_folder"])
            assert self._test_images_folder.exists(), "test images folder not found"

            self._test_xml_folder = Path(config.get("test_images_folder", self._test_images_folder))
            self._test_ds_name = "ds_test"

        # reading output folder
        if "output_folder" in config:
            self._out_folder = Path(config["output_folder"]).joinpath(self._model_name)
        else:
            self._out_folder = Path(os.getcwd()).joinpath(os.path.sep.join(["models", self._model_name]))
        self._out_folder.mkdir(exist_ok=True, parents=True)

        # reading labels maps
        labels_map = config.get("labels_map", None)
        if labels_map:
            if isinstance(labels_map, dict):
                self._labels_map = labels_map
            # elif isinstance(labels_map, str) and os.path.isfile(labels_map):
            #     self._labels_map = get_label_map_dict(labels_map)
            else:
                raise Exception("`labels map` parameter must be a dictionary or a file")

    def _register_datasets(self):
        self._dataset.register(self._train_ds_name, self._train_images_folder, self._train_xml_folder, "train")
        if self._test_images_folder is not None:
            self._dataset.register(self._test_ds_name, self._test_images_folder, self._test_xml_folder, "test")
