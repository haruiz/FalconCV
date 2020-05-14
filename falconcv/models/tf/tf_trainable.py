import functools
import itertools
import logging
import os
from pathlib import Path
import tensorflow as tf
from google.protobuf import text_format
from object_detection import model_hparams, model_lib, exporter
from object_detection.builders import graph_rewriter_builder, dataset_builder, model_builder
from object_detection.legacy import trainer
from object_detection.protos import pipeline_pb2
from object_detection.utils.label_map_util import get_label_map_dict
from sklearn.model_selection import train_test_split
from falconcv.models.api_model import ApiModel
from falconcv.decor import typeassert, pathassert
from falconcv.util import FileUtil
from .tf_model_zoo import TFModelZoo
from object_detection import export_tflite_ssd_graph_lib
import subprocess
import dask
from .util import Utilities
import typing

logger = logging.getLogger(__name__)


class TfTrainableModel(ApiModel):
    def __init__(self, config: dict):
        self._images = []
        # training parameters
        self._model_name = config.get("model", None)
        self._out_folder = config.get("output_folder", None)
        self._images_folder = config.get("images_folder", None)
        self._masks_folder = config.get("masks_folder", self._images_folder)
        self._xml_folder = config.get("xml_folder", self._images_folder)
        self._labels_map = config.get("labels_map", None)
        # pre-trained model paths
        self._checkpoint_model_folder = None
        self._checkpoint_model_pipeline_file = None
        # new model paths
        self._new_model_pipeline_file = os.path.join(self._out_folder, "pipeline.config")
        self._labels_map_file = os.path.join(self._out_folder, "label_map.pbtxt")
        self._val_record_file = os.path.join(self._out_folder, "val.record")
        self._train_record_file = os.path.join(self._out_folder, "train.record")
        self._new_model_pipeline_dict = None

        if isinstance(self._labels_map, dict):
            self._labels_map_dict = self._labels_map
            self._labels_map_dict = {k.title(): v for k, v in self._labels_map_dict.items()}
        elif isinstance(self._labels_map, str) and os.path.isfile(self._labels_map):
            self._labels_map_dict = get_label_map_dict(self._labels_map)
        else:
            raise Exception("Invalid labels map config parameter provided")

        assert Path(self._images_folder).exists(), "Images folder doesnt exist"

    def model_arch(self):
        pipeline_dict = Utilities.load_pipeline(self._checkpoint_model_pipeline_file)
        model_config = pipeline_dict["model"]  # read Detection model
        model_arch = model_config.WhichOneof("model")
        return model_arch

    def _load_dataset(self):
        exts = [".jpg", ".jpeg", ".png", ".xml"]
        files = Utilities.get_files(Path(self._images_folder), exts)
        files = sorted(files, key=lambda img: img.name)
        for img_name, img_files in itertools.groupby(files, key=lambda img: img.stem):
            img_file, xml_file, mask_file = None, None, None
            for file in img_files:
                if file.suffix in [".jpg", ".jpeg"]:
                    img_file = file
                elif file.suffix == ".xml":
                    xml_file = file
                elif file.suffix == ".png":
                    mask_file = file
            if img_file and xml_file:
                self._images.append({
                    "image": img_file,
                    "xml": xml_file,
                    "mask": mask_file
                })
        assert len(self._images), "Not images found at the folder {}".format(self._images_folder)

    def _mk_labels_map(self):
        if os.path.isfile(self._labels_map_file):
            os.remove(self._labels_map_file)
        with open(self._labels_map_file, 'a') as f:
            for name, idx in self._labels_map_dict.items():
                item = "item{{\n id: {} \n name: '{}'\n}} \n".format(idx, name.title())
                f.write(item)

    @classmethod
    def _mk_record_file(cls, images: dict, out_file, labels_map):
        delayed_tasks = [dask.delayed(Utilities.image_to_example)(img["image"], img["xml"], img["mask"], labels_map) for
                         img in images]
        examples = dask.compute(*delayed_tasks)
        with tf.python_io.TFRecordWriter(out_file) as writer:
            for tf_example in examples:
                writer.write(tf_example.SerializeToString())

    def _mk_records(self, split_size):
        train_images, val_images = train_test_split(self._images, test_size=split_size, random_state=42, shuffle=True)
        self._mk_record_file(train_images, self._train_record_file, self._labels_map_dict)
        self._mk_record_file(val_images, self._val_record_file, self._labels_map_dict)

    @typeassert(epochs=int, val_split=float, override_pipeline=bool, clear_folder=bool)
    def train_and_eval(self, epochs=100, val_split=0.3, clear_folder=False, override_pipeline=False):
        try:
            if clear_folder:
                FileUtil.clear_folder(self._out_folder)
            self._mk_labels_map()
            self._mk_records(val_split)
            if not os.path.isfile(self._new_model_pipeline_file) or override_pipeline:
                pipeline = Utilities.load_pipeline(self._checkpoint_model_pipeline_file)
                num_classes = len(self._labels_map_dict)
                pipeline = Utilities.update_pipeline(
                    pipeline,
                    num_classes,
                    self._checkpoint_model_folder,
                    self._labels_map_file,
                    self._val_record_file,
                    self._train_record_file,
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
        return self

    @typeassert(checkpoint=int, out_folder=str)
    def freeze(self, checkpoint, out_folder: typing.Union[str, Path] = None):
        try:
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

    @typeassert(checkpoint=int, out_folder=str, add_postprocessing_op=bool, use_regular_nms=bool,
                max_classes_per_detection=int)
    def to_tflite(self, checkpoint,
                  out_folder=None,
                  input_size=(300, 300),
                  max_detections=10,
                  add_postprocessing_op=True,
                  use_regular_nms=True,
                  max_classes_per_detection=1):
        try:
            assert self.model_arch() == "ssd", "This method is only supported for ssd models"
            model_checkpoint = "{}/model.ckpt-{}".format(self._out_folder, checkpoint)
            out_folder = out_folder if out_folder else self._out_folder
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.gfile.GFile(self._new_model_pipeline_file, 'r') as f:
                text_format.Merge(f.read(), pipeline_config)
            export_tflite_ssd_graph_lib.export_tflite_graph(
                pipeline_config,
                model_checkpoint,
                out_folder,
                add_postprocessing_op,
                max_detections,
                max_classes_per_detection, use_regular_nms=use_regular_nms)
            # convert to tflite
            cmd = '''toco
               --output_format=TFLITE
               --graph_def_file="{}"
               --output_file="{}"
               --input_shapes="1,{},{},3"
               --input_arrays=normalized_input_image_tensor
               --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3
               --inference_type=FLOAT --allow_custom_ops
               ''' \
                .format(
                Path(out_folder).joinpath("tflite_graph.pb"),
                Path(out_folder).joinpath("model.tflite"),
                input_size[0], input_size[1]
            )
            cmd = " ".join([line.strip() for line in cmd.splitlines()])
            print(subprocess.check_output(cmd, shell=True).decode())
            return self
        except Exception as ex:
            raise Exception("Error converting the model {}".format(ex)) from ex

    def eval(self, *args, **kwargs):
        return super(TfTrainableModel, self).eval()

    def __enter__(self):
        try:
            os.makedirs(self._out_folder, exist_ok=True)
            self._checkpoint_model_folder = TFModelZoo.download_model(self._model_name)
            self._checkpoint_model_pipeline_file = TFModelZoo.download_pipeline(self._model_name)
            self._load_dataset()
            return self
        except  Exception as ex:
            raise Exception("Error loading the model : {}".format(ex)) from ex

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Error loading the model:  {}, {}".format(exc_type, str(exc_val)))
            raise exc_val

    @tf.contrib.framework.deprecated(None, 'Use object_detection/model_main.py.')
    def train(self, epochs=100, val_split=0.3, clear_folder=False, override_pipeline=False):
        try:
            tf.logging.set_verbosity(tf.logging.INFO)
            tf.disable_eager_execution()
            if clear_folder:
                FileUtil.clear_folder(self._out_folder)
            self._mk_labels_map()
            self._mk_records(val_split)
            if not os.path.isfile(self._new_model_pipeline_file) or override_pipeline:
                pipeline = Utilities.load_pipeline(self._checkpoint_model_pipeline_file)
                num_classes = len(self._labels_map_dict)
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
        return super(TfTrainableModel, self).train()
