import functools
import logging
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from google.protobuf import text_format
from object_detection import model_hparams, model_lib, exporter
from object_detection.builders import graph_rewriter_builder, dataset_builder, model_builder
from object_detection.legacy import trainer
from object_detection.protos import pipeline_pb2
from object_detection.utils.label_map_util import get_label_map_dict
from sklearn.model_selection import train_test_split
from falconcv.models.api_model import ApiModel
from falconcv.decor import typeassert
from falconcv.ds import read_pascal_dataset
from falconcv.util import FileUtil
from .zoo import ModelZoo
from .util import Utilities


logger = logging.getLogger(__name__)

class TfTrainableModel(ApiModel):
    def __init__(self, config: dict):
        self._model_name = config.get("model", None)
        self._out_folder = config.get("out_folder", None)
        self._images_folder = config.get("images_folder", None)
        self._masks_folder = config.get("masks_folder", self._images_folder)
        self._xml_folder = config.get("xml_folder", self._images_folder)
        assert self._images_folder, "image folder required"
        assert self._out_folder, "out folder required"
        assert self._model_name, "model name required"
        os.makedirs(self._out_folder, exist_ok=True)
        self._dataset_df = pd.DataFrame()
        self._checkpoint_model_folder = None
        self._checkpoint_model_pipeline_file = None

        self._new_model_pipeline_file = os.path.join(self._out_folder, "pipeline.config")
        self._labels_map_file = os.path.join(self._out_folder, "label_map.pbtx")
        self._val_record_file = os.path.join(self._out_folder, "val.record")
        self._train_record_file = os.path.join(self._out_folder, "train.record")
        self._new_model_pipeline_dict = None
        self._labels_map_dict = None

    def labels(self):
        return self._dataset_df["class"].unique()

    def _load_dataset(self):
        self._dataset_df = read_pascal_dataset(self._xml_folder)

    def _mk_labels_map(self):
        if os.path.isfile(self._labels_map_file):
            os.remove(self._labels_map_file)
        with open(self._labels_map_file, 'a') as f:
            for idx, name in enumerate(self.labels()):
                item = "item{{\n id: {} \n name: '{}'\n}} \n".format(idx + 1, name)
                f.write(item)

    @classmethod
    def _mk_record_file(cls, df: pd.DataFrame, out_file, map_labels_file):
        map_labels = get_label_map_dict(map_labels_file)
        with tf.python_io.TFRecordWriter(out_file) as writer:
            for key, rows in df.groupby(['image_path']):
                tf_example = Utilities.create_record(key, rows, map_labels)
                writer.write(tf_example.SerializeToString())

    def _mk_records(self, split_size):
        train_df, val_df = train_test_split(self._dataset_df, test_size=split_size, random_state=42, shuffle=True)
        self._mk_record_file(train_df, self._train_record_file, self._labels_map_file)
        self._mk_record_file(val_df, self._val_record_file, self._labels_map_file)

    @typeassert(epochs=int, val_split=float, override_pipeline=bool, clear_folder=bool)
    def train(self, epochs=100, val_split=0.3, clear_folder=False, override_pipeline=False):
        try:
            if clear_folder:
                FileUtil.clear_folder(self._out_folder)
            assert not self._dataset_df.empty, "the dataset is empty, or the model wasn't initialized correctly"
            self._mk_labels_map()
            self._mk_records(val_split)
            if not os.path.isfile(self._new_model_pipeline_file) or override_pipeline:
                pipeline = Utilities.load_pipeline(self._checkpoint_model_pipeline_file)
                num_classes = len(self.labels())
                pipeline = Utilities.update_pipeline(
                    pipeline,
                    num_classes,
                    str(Path(self._checkpoint_model_folder)),
                    str(Path(self._labels_map_file)),
                    str(Path(self._val_record_file)),
                    str(Path(self._train_record_file)),
                    epochs
                )
                Utilities.save_pipeline(pipeline, self._out_folder)
            os.makedirs(os.path.join(self._out_folder, "export/Servo"), exist_ok=True)
            # training
            tf.logging.set_verbosity(tf.logging.INFO)
            # device_name=tf.test.gpu_device_name()
            # if device_name != '/device:GPU:0':
            gpu_available = tf.test.is_gpu_available()
            session_config = tf.ConfigProto()
            if gpu_available:
                config = tf.estimator.RunConfig(
                    model_dir=self._out_folder)
            else:
                session_config.gpu_options.allow_growth = True
                config = tf.estimator.RunConfig(
                    model_dir=self._out_folder,
                    session_config=session_config)

            train_and_eval_dict = model_lib.create_estimator_and_inputs(
                run_config=config,
                hparams=model_hparams.create_hparams(None),
                pipeline_config_path=self._new_model_pipeline_file,
                train_steps=epochs,
                sample_1_of_n_eval_examples=1,
                sample_1_of_n_eval_on_train_examples=5)
            estimator = train_and_eval_dict['estimator']
            train_input_fn = train_and_eval_dict['train_input_fn']
            eval_input_fns = train_and_eval_dict['eval_input_fns']
            eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
            predict_input_fn = train_and_eval_dict['predict_input_fn']
            train_steps = train_and_eval_dict['train_steps']
            train_spec, eval_specs = model_lib.create_train_and_eval_specs(
                train_input_fn,
                eval_input_fns,
                eval_on_train_input_fn,
                predict_input_fn,
                train_steps,
                eval_on_train_data=False)
            # Currently only a single Eval Spec is allowed.
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
        except Exception as ex:
            raise Exception("Error training the model {}".format(ex)) from ex
        return super(TfTrainableModel, self).train()

    @typeassert(checkpoint=int, out_folder=str)
    def freeze(self, checkpoint, out_folder=None):
        try:
            tf.disable_eager_execution()
            model_checkpoint = "{}/model.ckpt-{}".format(self._out_folder, checkpoint)
            if out_folder:
                frozen_model_path = os.path.join(out_folder, "frozen_inference_graph.pb")
            else:
                frozen_model_path = os.path.join(self._out_folder, "export/frozen_inference_graph.pb")
            model_pipeline = self._new_model_pipeline_file
            frozen_model_dir = os.path.dirname(frozen_model_path)
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            tf.reset_default_graph()
            with tf.gfile.GFile(model_pipeline, 'r') as f:
                text_format.Merge(f.read(), pipeline_config)
            exporter.export_inference_graph(
                input_type="image_tensor",
                pipeline_config=pipeline_config,
                trained_checkpoint_prefix=model_checkpoint,
                output_directory=frozen_model_dir,
                input_shape=None,
                write_inference_graph=False)
        except Exception as ex:
            raise Exception("Error freezing the model {}".format(ex)) from ex
        return super(TfTrainableModel, self).freeze()

    def eval(self, *args, **kwargs):
        return super(TfTrainableModel, self).eval()

    def __enter__(self):
        try:
            self._checkpoint_model_folder = ModelZoo.download_model(self._model_name)
            self._checkpoint_model_pipeline_file = ModelZoo.download_pipeline(self._model_name)
            self._load_dataset()
            return self
        except  Exception as ex:
            raise Exception("Error loading the model {}".format(ex)) from ex


    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Error loading the model:  {}, {}".format(exc_type, str(exc_val)))


    @tf.contrib.framework.deprecated(None, 'Use object_detection/model_main.py.')
    def _train(self, epochs=100, val_split=0.3, clear_folder=False, override_pipeline=False):
        try:
            tf.logging.set_verbosity(tf.logging.INFO)
            tf.disable_eager_execution()
            if clear_folder:
                FileUtil.clear_folder(self._out_folder)
            assert not self._dataset_df.empty, "the dataset is empty, or the model wasn't initialized correctly"
            self._mk_labels_map()
            self._mk_records(val_split)
            if not os.path.isfile(self._new_model_pipeline_file) or override_pipeline:
                pipeline = Utilities.load_pipeline(self._checkpoint_model_pipeline_file)
                num_classes = len(self.labels())
                pipeline = Utilities.update_pipeline(
                    pipeline,
                    num_classes,
                    str(Path(self._checkpoint_model_folder)),
                    str(Path(self._labels_map_file)),
                    str(Path(self._val_record_file)),
                    str(Path(self._train_record_file)),
                    epochs
                )
                Utilities.save_pipeline(pipeline, self._out_folder)
            ps_tasks = 0
            worker_replicas = 1
            worker_job_name = 'lonely_worker'
            task = 0
            is_chief = True
            master = ''
            graph_rewriter_fn = None
            # loading and reading  the config file
            configs = Utilities.load_pipeline(self._new_model_pipeline_file)
            model_config = configs['model']
            train_config = configs['train_config']
            input_config = configs['train_input_config']
            # creating the tf object detection api model (from the config parameters)
            model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=True)
            def get_next(config):
                return dataset_builder.make_initializable_iterator(dataset_builder.build(config)).get_next()
            create_input_dict_fn = functools.partial(get_next, input_config)
            if 'graph_rewriter_config' in configs:
                graph_rewriter_fn = graph_rewriter_builder.build(configs['graph_rewriter_config'], is_training=True)
            # training the model with the new parameters
            trainer.train(
                create_input_dict_fn,
                model_fn,
                train_config,
                master,
                task,
                1,
                worker_replicas,
                False,
                ps_tasks,
                worker_job_name,
                is_chief,
                self._out_folder,
                graph_hook_fn=graph_rewriter_fn)
        except Exception as ex:
            raise Exception("Error training the model : {}".format(ex)) from ex
