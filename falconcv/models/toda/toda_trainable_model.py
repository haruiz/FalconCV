import os
import typing
from pathlib import Path

import numpy as np

from falconcv.data import PascalVOCDataset
from falconcv.decor import requires, typeassert, exception
from falconcv.models.api_model import ApiModel
from falconcv.models.toda import TODAZoo, model_lib_v2
from falconcv.models.toda.toda_pipeline_handler import PretrainedModelPipelineHandler
from falconcv.util import FileUtils, LibUtils, TODAUtils
from .toda_trained_model import TODATrainedModel

try:

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    import tensorflow.compat.v1 as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
except ImportError:
    ...
import logging

logger = logging.getLogger("rich")


@requires("object_detection")
class TODATrainableModel(ApiModel):
    @typeassert
    def __init__(
        self,
        pretrained_model_path: typing.Union[str, Path],
        pretrained_model_config_path: typing.Union[str, Path],
        images_folder_path: typing.Union[str, Path],
        output_folder_path: typing.Union[None, str, Path] = None,
    ):
        # pretrained models path
        self._pretrained_model_path = Path(pretrained_model_path)
        self._pretrained_model_checkpoint_path = self._pretrained_model_path.joinpath(
            "checkpoint/ckpt-0"
        )
        self._pretrained_model_config_path = Path(pretrained_model_config_path)

        # new model paths
        self._new_model_output_folder_path = Path(output_folder_path)
        self._new_model_pipeline_file_path = (
            self._new_model_output_folder_path.joinpath("pipeline.config")
        )
        self._new_model_labels_map_path = self._new_model_output_folder_path.joinpath(
            "label_map.pbtxt"
        )
        self._new_model_checkpoint_folder = self._new_model_output_folder_path.joinpath(
            "checkpoint"
        )
        self._new_model_export_folder = self._new_model_output_folder_path.joinpath(
            "export"
        )
        self._new_model_images_folder_path = images_folder_path
        self._new_model_val_record_path = self._new_model_output_folder_path.joinpath(
            "val.record"
        )
        self._new_model_train_record_path = self._new_model_output_folder_path.joinpath(
            "train.record"
        )

        self._dataset = None
        self._pipeline = None
        self._label_map = None

    @classmethod
    def from_config(cls, config: dict):
        assert isinstance(config, dict), "config must be a dict"
        checkpoint_uri = config.get("checkpoint_uri", None)
        pipeline_uri = config.get("pipeline_uri", None)
        assert (
            checkpoint_uri or pipeline_uri
        ), "checkpoint_uri or pipeline_uri must be provided"
        images_folder = config.get("images_folder", None)
        assert images_folder, "images_folder must be provided"

        default_model_folder = Path(__file__).parent
        output_folder = config.get("output_folder", default_model_folder)
        folder_name = FileUtils.get_file_name_from_uri(checkpoint_uri)
        output_folder = Path(output_folder).joinpath("models").joinpath(folder_name)
        output_folder.mkdir(parents=True, exist_ok=True)

        checkpoint_path = TODAZoo.download_model_checkpoint(
            checkpoint_uri, LibUtils.models_home("toda")
        )
        pipeline_path = TODAZoo.download_model_pipeline(
            pipeline_uri, LibUtils.pipelines_home("toda")
        )
        return cls(checkpoint_path, pipeline_path, images_folder, output_folder)

    def __enter__(self):
        # load dataset
        self._mk_output_folders()
        if self._new_model_pipeline_file_path.exists():
            self._pipeline = PretrainedModelPipelineHandler(
                self._new_model_pipeline_file_path
            )
        else:
            self._pipeline = PretrainedModelPipelineHandler(
                self._pretrained_model_config_path
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Error loading the model:  {exc_type}, {exc_val}")
            raise exc_val

    @property
    def pipeline(self) -> PretrainedModelPipelineHandler:
        return self._pipeline

    def _mk_output_folders(self):
        if self._new_model_output_folder_path is None:
            self._new_model_output_folder_path = Path(__file__).parent.joinpath("model")
        self._new_model_output_folder_path.mkdir(exist_ok=True, parents=True)
        self._new_model_checkpoint_folder.mkdir(exist_ok=True, parents=True)
        self._new_model_export_folder.mkdir(exist_ok=True, parents=True)

    def _mk_labels_map_file(self):
        # create labels map file
        self._dataset.mk_labels_map_file(self._new_model_labels_map_path)

    def _mk_pipeline_file(self, batch_size, num_steps, use_tpu):
        # create pipeline file
        self.pipeline.num_classes = len(self._dataset.labels)
        self.pipeline.batch_size = batch_size
        self.pipeline.num_steps = num_steps
        self.pipeline.fine_tune_checkpoint_type = "detection"
        self.pipeline.use_bfloat16 = use_tpu
        self.pipeline.set_config_paths(
            self._new_model_train_record_path,
            self._new_model_val_record_path,
            self._new_model_labels_map_path,
            self._pretrained_model_checkpoint_path,
        )
        print(self.pipeline.input_size)
        self.pipeline.save(self._new_model_output_folder_path)

    def _mk_records_files(self, random_seed=None, ratio=0.8, shuffle=True):
        # create records files
        training_set, validation_set = self._dataset.split(
            ratio, shuffle=shuffle, random_state=random_seed
        )
        training_set.mk_record_file(self._new_model_train_record_path)
        validation_set.mk_record_file(self._new_model_val_record_path)
        logger.info(
            f"Training images {len(training_set)}, Validation images {len(validation_set)}"
        )

    def _get_training_strategy(self, num_workers, tpu_name, use_tpu):
        if use_tpu:
            # TPU is automatically inferred if tpu_name is None and
            # we are running under cloud ai-platform.
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
        elif num_workers > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()
        return strategy

    @exception
    def train(
        self,
        epochs=100,
        ratio=0.8,
        batch_size=32,
        random_seed=42,
        shuffle=True,
        clear_folder=True,
        use_tpu=False,
        tpu_name=None,
        num_workers=1,
        checkpoint_every_n=500,
    ):
        if clear_folder:
            FileUtils.clear_folder(self._new_model_checkpoint_folder)
        # load dataset
        self._dataset = PascalVOCDataset.from_folder(self._new_model_images_folder_path)
        logger.info(f"{len(self._dataset)} images found in the dataset")
        logger.info(f"{self._dataset.labels} labels found in the dataset")
        strategy = self._get_training_strategy(num_workers, tpu_name, use_tpu)

        # create required files
        self._mk_records_files(random_seed, ratio, shuffle)
        self._mk_labels_map_file()
        self._mk_pipeline_file(batch_size, epochs, use_tpu)

        # starts training
        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=str(self._new_model_pipeline_file_path),
                model_dir=self._new_model_checkpoint_folder,
                train_steps=epochs,
                use_tpu=use_tpu,
                checkpoint_every_n=checkpoint_every_n,
                record_summaries=True,
            )

    @exception
    def evaluate(
        self,
        num_train_steps=None,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=5,
        eval_timeout=3600,
    ):
        model_lib_v2.eval_continuously(
            pipeline_config_path=self._new_model_pipeline_file_path,
            model_dir=self._new_model_checkpoint_folder,
            train_steps=num_train_steps,
            sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(sample_1_of_n_eval_on_train_examples),
            checkpoint_dir=self._new_model_checkpoint_folder,
            wait_interval=300,
            timeout=eval_timeout,
        )

    @exception
    def to_saved_model(self, **kwargs):

        out_folder = self._new_model_export_folder.joinpath("server")
        model_pipeline_path = self._new_model_pipeline_file_path
        model_checkpoint_path = self._new_model_checkpoint_folder

        TODAUtils.export_saved_model(
            model_checkpoint_path, model_pipeline_path, out_folder, **kwargs
        )

    def to_tflite(self, **kwargs):
        model_pipeline_path = self._new_model_pipeline_file_path
        model_checkpoint = self._new_model_checkpoint_folder
        model_labels_map_path = self._new_model_labels_map_path
        tflite_model_output_folder = self._new_model_export_folder.joinpath("tflite")

        TODAUtils.export_tf_lite(
            model_checkpoint,
            model_pipeline_path,
            model_labels_map_path,
            tflite_model_output_folder,
            **kwargs,
        )

    @exception
    def __call__(
        self,
        input_image: typing.Union[str, Path, np.ndarray],
        size: typing.Union[None, typing.Tuple[int, int]] = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        top_k: int = 10,
    ):
        saved_model_folder = self._new_model_export_folder.joinpath(
            "server/saved_model"
        )
        assert saved_model_folder.exists(), "Saved model folder does not exist"
        return TODATrainedModel(saved_model_folder, self._new_model_labels_map_path)(
            input_image, size, threshold, mask_threshold, top_k
        )
