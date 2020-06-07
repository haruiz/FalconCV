import functools
import itertools
import logging
import os
import subprocess
from pathlib import Path

import tensorflow as tf

tf_logger = tf.get_logger()
tf_logger.propagate = False
from object_detection import model_hparams, model_lib, export_tflite_ssd_graph_lib, exporter
from object_detection.builders import graph_rewriter_builder, dataset_builder, model_builder
from object_detection.legacy import trainer
from object_detection.utils.config_util import create_pipeline_proto_from_configs, get_configs_from_pipeline_file, \
    save_pipeline_config, create_configs_from_pipeline_proto, update_input_reader_config, _update_tf_record_input_path, \
    merge_external_params_with_configs
from object_detection.utils.label_map_util import get_label_map_dict

from falconcv.decor import typeassert
from falconcv.models.api_model import ApiModel
from falconcv.util import FileUtil
from .zoo import ModelZoo
from .util import Utilities

import dask
import typing

from .pascal_voc_ds import PascalVOCImage, PascalVOCDataset

logger = logging.getLogger(__name__)


class TfTrainableModel(ApiModel):
    def __init__(self, config: dict):
        self._dataset = PascalVOCDataset()
        # check training parameters
        assert "model" in config and isinstance(config["model"],
                                                str), "`model` parameter is required, and must be an string"
        # check images folder
        assert "images_folder" in config and isinstance(config["images_folder"],
                                                        str), "`images folder` parameter is required, and must be an string"
        self._images_folder = Path(config["images_folder"])
        assert self._images_folder.exists(), "images folder not found"

        # reading config
        self._model_name = config["model"]
        self._masks_folder = Path(config.get("masks_folder", self._images_folder))
        self._xml_folder = Path(config.get("xml_folder", self._images_folder))

        # pre-trained model paths
        self._checkpoint_model_folder: Path = Path()
        self._checkpoint_model_pipeline_file: Path = Path()

        # check output folder
        if "output_folder" in config:
            assert isinstance(config["output_folder"], str), "`output_folder` must be an string"
            self._out_folder = Path(config["output_folder"]).joinpath(self._model_name)
        else:
            self._out_folder = Path(os.getcwd()).joinpath(os.path.sep.join(["models", self._model_name]))
        self._out_folder.mkdir(exist_ok=True, parents=True)

        # model attributes
        self._pipeline = None
        self._labels_map = None
        # reading label map
        labels_map = config.get("labels_map", None)
        if labels_map:
            if isinstance(labels_map, dict):
                self._labels_map = labels_map
            elif isinstance(labels_map, str) and os.path.isfile(labels_map):
                self._labels_map = get_label_map_dict(labels_map)
            else:
                raise Exception("`labels map` parameter must be a dictionary or a file")

        # new model paths
        self._pipeline_file = self._out_folder.joinpath("pipeline.config")
        self._labels_map_file = self._out_folder.joinpath("label_map.pbtxt")
        self._val_record_file = self._out_folder.joinpath("val.record")
        self._train_record_file = self._out_folder.joinpath("train.record")

    @property
    def pipeline(self):
        if self._pipeline is None:
            raise Exception("Model not initialized correctly")
        return self._pipeline

    @property
    def arch(self):
        return self.pipeline.model.WhichOneof("model")

    @property
    def num_classes(self):
        return getattr(self.pipeline.model, self.arch).num_classes

    @property
    def input_size(self):
        resizer_config = getattr(self.pipeline.model, self.arch).image_resizer
        if resizer_config.HasField("fixed_shape_resizer"):
            return [
                resizer_config.fixed_shape_resizer.width,
                resizer_config.fixed_shape_resizer.height
            ]
        elif resizer_config.HasField("keep_aspect_ratio_resizer"):
            return [
                resizer_config.keep_aspect_ratio_resizer.min_dimension,
                resizer_config.keep_aspect_ratio_resizer.max_dimension
            ]
        elif resizer_config.HasField("identity_resizer") or resizer_config.HasField("conditional_shape_resizer"):
            return [-1, -1]
        else:
            raise ValueError("Unknown image resizer type.")

    @property
    def resizer_type(self):
        resizer_config = getattr(self.pipeline.model, self.arch).image_resizer
        if resizer_config.HasField("fixed_shape_resizer"):
            return "fixed_shape_resizer"
        elif resizer_config.HasField("keep_aspect_ratio_resizer"):
            return "keep_aspect_ratio_resizer"
        elif resizer_config.HasField("identity_resizer"):
            return "identity_resizer"
        elif resizer_config.HasField("conditional_shape_resizer"):
            return "conditional_shape_resizer"
        else:
            raise ValueError("Unknown image resizer type.")

    @input_size.setter
    def input_size(self, value):
        assert isinstance(value, tuple), "invalid input size"
        resizer_config = getattr(self.pipeline.model, self.arch).image_resizer
        if resizer_config.HasField("fixed_shape_resizer"):
            resizer_config.fixed_shape_resizer.width = value[0]
            resizer_config.fixed_shape_resizer.height = value[1]
        elif resizer_config.HasField("keep_aspect_ratio_resizer"):
            resizer_config.keep_aspect_ratio_resizer.min_dimension = value[0]
            resizer_config.keep_aspect_ratio_resizer.max_dimension = value[1]

    @property
    def num_steps(self):
        return self.pipeline.train_config.num_steps

    @num_steps.setter
    def num_steps(self, value):
        assert isinstance(value, int), "invalid value"
        self.pipeline.train_config.num_steps = value

    @property
    def batch_size(self):
        return self.pipeline.train_config.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.pipeline.train_config.batch_size = value

    @typeassert(checkpoint=int, out_folder=str)
    def freeze(self, checkpoint, out_folder: typing.Union[str, Path] = None):
        try:
            if out_folder:
                frozen_model_dir = Path(out_folder)
            else:
                frozen_model_dir = self._out_folder.joinpath("export")
            model_checkpoint = self._out_folder.joinpath("model.ckpt-{}".format(checkpoint))
            exporter.export_inference_graph(
                input_type="image_tensor",
                pipeline_config=self._pipeline,
                trained_checkpoint_prefix=str(model_checkpoint),
                output_directory=str(frozen_model_dir),
                input_shape=None,
                write_inference_graph=False)
        except Exception as ex:
            raise Exception("Error freezing the model {}".format(ex)) from ex
        return super(TfTrainableModel, self).freeze()

    def eval(self, *args, **kwargs):
        return super(TfTrainableModel, self).eval()

    @property
    def labels_map(self):
        return self._labels_map

    def _mk_labels_map(self):
        if self._labels_map_file.exists():
            self._labels_map_file.unlink()
        with open(str(self._labels_map_file), 'a') as f:
            for name, idx in self._labels_map.items():
                item = "item{{\n id: {} \n name: '{}'\n}} \n".format(idx, name.title())
                f.write(item)

    @classmethod
    def _mk_record_file(cls, images: dict, out_file, labels_map):
        delayed_tasks = [dask.delayed(img.to_example_record)(labels_map) for img in images]
        examples = dask.compute(*delayed_tasks)
        with tf.io.TFRecordWriter(str(out_file)) as writer:
            for tf_example in examples:
                if tf_example:
                    writer.write(tf_example.SerializeToString())

    def _mk_records(self, split_size):
        train_images, val_images = self._dataset.split(split_size)
        self._mk_record_file(train_images, self._train_record_file, self._labels_map)
        self._mk_record_file(val_images, self._val_record_file, self._labels_map)

    def _load_dataset(self):
        img_files = FileUtil.get_files(self._images_folder, [".jpg", ".jpeg"])
        xml_files = FileUtil.get_files(self._xml_folder, [".xml"])
        mask_files = FileUtil.get_files(self._xml_folder, [".png"])
        files = img_files + xml_files + mask_files
        files = sorted(files, key=lambda img: img.stem)
        images_files = []
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
                images_files.append(PascalVOCImage(
                    img_path=img_file,
                    xml_Path=xml_file,
                    mask_Path=mask_file
                ))
        self._dataset.load(images_files, batch_size=200)
        assert not self._dataset.empty, "Not images found at the folder {}".format(self._images_folder)
        # generate labels map if is not provided
        if self._labels_map is None:
            labels = set()
            for img in self._dataset:
                for bounding_box in img.annotations["object"]:
                    labels.add(bounding_box["name"].strip().title())
            self._labels_map = {l: i + 1 for i, l in enumerate(labels)}
        # update number of classes
        getattr(self.pipeline.model, self.arch).num_classes = len(self._labels_map)

    def _set_config_paths(self):
        configs = create_configs_from_pipeline_proto(self.pipeline)
        update_input_reader_config(configs,
                                   key_name="train_input_config",
                                   input_name=None,
                                   field_name="input_path",
                                   value=str(self._val_record_file),
                                   path_updater=_update_tf_record_input_path)
        update_input_reader_config(configs,
                                   key_name="eval_input_configs",
                                   input_name=None,
                                   field_name="input_path",
                                   value=str(self._train_record_file),
                                   path_updater=_update_tf_record_input_path)
        update_dict = {
            "label_map_path": str(self._labels_map_file),
            "train_config.fine_tune_checkpoint": str(self._checkpoint_model_folder.joinpath("model.ckpt"))
        }
        configs = merge_external_params_with_configs(configs, kwargs_dict=update_dict)
        self._pipeline = create_pipeline_proto_from_configs(configs)

    def __enter__(self):
        try:
            self._checkpoint_model_folder = ModelZoo.download_model(self._model_name)
            self._checkpoint_model_pipeline_file = ModelZoo.download_pipeline(self._model_name)
            # load pipeline
            if self._pipeline_file.exists():
                configs = get_configs_from_pipeline_file(str(self._pipeline_file))  # load config as a dict
            else:
                configs = get_configs_from_pipeline_file(
                    str(self._checkpoint_model_pipeline_file))  # load config as a dict
            self._pipeline = create_pipeline_proto_from_configs(configs)  # convert to a protobuffer
            # load dataset
            self._load_dataset()
            self._set_config_paths()
            return self
        except Exception as ex:
            raise Exception("Error loading the model : {}".format(ex)) from ex

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Error loading the model:  {}, {}".format(exc_type, str(exc_val)))
            raise exc_val

    @tf.contrib.framework.deprecated(None, 'Use object_detection/model_main.py.')
    def train(self, epochs=100, val_split=0.3, clear_folder=False, override_pipeline=False, eval=False):
        try:

            if clear_folder:
                FileUtil.clear_folder(self._out_folder)
            self.num_steps = epochs
            self._mk_labels_map()
            self._mk_records(val_split)
            # update pipeline
            self._out_folder.joinpath(os.path.sep.join(["export", "Servo"])).mkdir(exist_ok=True, parents=True)
            # merge pipelines
            save_pipeline_config(self.pipeline, str(self._out_folder))
            # start training
            tf.logging.set_verbosity(tf.logging.INFO)
            if eval:
                self._train_and_eval()
            else:
                self._train()
        except Exception as ex:
            raise Exception("Error training the model : {}".format(ex)) from ex
        return super(TfTrainableModel, self).train()

    def _train_and_eval(self):
        gpu_available = tf.test.is_gpu_available()
        session_config = tf.ConfigProto()
        if gpu_available:
            run_config = tf.estimator.RunConfig(
                model_dir=str(self._out_folder))
        else:
            session_config.gpu_options.allow_growth = True
            run_config = tf.estimator.RunConfig(
                model_dir=str(self._out_folder),
                session_config=session_config)
        train_and_eval_dict = model_lib.create_estimator_and_inputs(
            run_config=run_config,
            hparams=model_hparams.create_hparams(None),
            pipeline_config_path=str(self._pipeline_file),
            train_steps=self.num_steps,
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
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

    def _train(self):
        tf.disable_eager_execution()
        ps_tasks = 0
        worker_replicas = 1
        worker_job_name = 'lonely_worker'
        task = 0
        is_chief = True
        master = ''
        graph_rewriter_fn = None
        # loading and reading  the config file
        configs = create_configs_from_pipeline_proto(self.pipeline)
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
        trainer.train(create_input_dict_fn, model_fn, train_config, master, task, 1, worker_replicas, False, ps_tasks,
                      worker_job_name, is_chief, str(self._out_folder), graph_hook_fn=graph_rewriter_fn)

    @typeassert(checkpoint=int, out_folder=str, add_postprocessing_op=bool, use_regular_nms=bool,
                max_classes_per_detection=int)
    def to_tflite(self, checkpoint,
                  out_folder=None,
                  max_detections=10,
                  add_postprocessing_op=True,
                  use_regular_nms=True,
                  max_classes_per_detection=1):
        try:
            assert self.arch == "ssd", "This method is only supported for ssd models"
            model_checkpoint = str(self._out_folder.joinpath("model.ckpt-{}".format(checkpoint)))
            tflite_model_folder = Path(out_folder) if out_folder else self._out_folder

            export_tflite_ssd_graph_lib.export_tflite_graph(
                self._pipeline,
                model_checkpoint,
                str(tflite_model_folder),
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
                tflite_model_folder.joinpath("tflite_graph.pb"),
                tflite_model_folder.joinpath("model.tflite"),
                self.input_size[0], self.input_size[1]
            )
            cmd = " ".join([line.strip() for line in cmd.splitlines()])
            print(subprocess.check_output(cmd, shell=True).decode())
            return self
        except Exception as ex:
            raise Exception("Error converting the model {}".format(ex)) from ex

    @typeassert(out_folder=str, device=str, force=bool)
    def to_OpenVINO(self, out_folder=None, device="CPU", pre_trained=False, force=False):
        try:
            assert "INTEL_OPENVINO_DIR" in os.environ, "OpenVINO workspace not initialized"
            OpenVINO_dir = Path(os.environ["INTEL_OPENVINO_DIR"])
            out_folder = out_folder if out_folder else self._out_folder
            # define data type
            data_type = "FP16" if device == "MYRIAD" else "FP32"
            model_arch = self._model_name.split("_")[0]
            front_openvino_file = Utilities.get_openvino_front_file(model_arch)
            logger.info("[INFO] front file picked:  {}".format(front_openvino_file))
            # define front file
            front_openvino_file = r"deployment_tools/model_optimizer/extensions/front/tf/{}".format(front_openvino_file)
            front_openvino_file = OpenVINO_dir.joinpath(front_openvino_file)

            # define output files
            xml_file = out_folder.joinpath("frozen_inference_graph.xml")
            bin_file = out_folder.joinpath("frozen_inference_graph.bin")
            if not os.path.exists(xml_file) or not os.path.exists(bin_file) or force:
                model_config = self._checkpoint_model_pipeline_file if pre_trained else self._pipeline_file
                optimizer_script = "deployment_tools/model_optimizer/mo_tf.py"
                optimizer_script = OpenVINO_dir.joinpath(optimizer_script)
                frozen_model = list(self._out_folder.rglob("**/frozen_inference_graph.pb"))[0]
                cmd = '''python "{}" 
                                --input_model "{}" 
                                --transformations_config "{}" 
                                --output_dir "{}"   
                                --data_type {}                            
                                --reverse_input_channels                                
                                --tensorflow_object_detection_api_pipeline_config "{}"                
                                    ''' \
                    .format(optimizer_script,
                            frozen_model,
                            front_openvino_file,
                            out_folder,
                            data_type,
                            model_config
                            )
                cmd = " ".join([line.strip() for line in cmd.splitlines()])
                print(subprocess.check_output(cmd, shell=True).decode())
        except Exception as ex:
            raise Exception("Error converting the model {}".format(ex)) from ex
