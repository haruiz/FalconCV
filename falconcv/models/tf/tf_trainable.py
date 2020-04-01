import logging
import os
from pathlib import Path

import pandas as pd



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from google.protobuf import text_format
from object_detection import model_hparams,model_lib,exporter
from object_detection.protos import pipeline_pb2
from object_detection.utils.label_map_util import get_label_map_dict
from sklearn.model_selection import train_test_split
from falconcv.models.api_model import ApiModel
from falconcv.decor import typeassert
from falconcv.ds import read_pascal_dataset

from falconcv.models.tf.util import Utilities
from falconcv.util import FileUtil
logger=logging.getLogger(__name__)


class TfTrainableModel(ApiModel):
    def __init__(self,config: dict):
        self._model_name=config.get("model",None)
        self._out_folder=config.get("out_folder",None)
        self._images_folder=config.get("images_folder",None)
        self._masks_folder=config.get("masks_folder",self._images_folder)
        self._xml_folder=config.get("xml_folder",self._images_folder)
        assert self._images_folder,"image folder required"
        assert self._out_folder,"out folder required"
        assert self._model_name,"model name required"
        os.makedirs(self._out_folder,exist_ok=True)

        self._dataset_df=pd.DataFrame()

        self._checkpoint_model_folder=None
        self._checkpoint_model_pipeline_file=None

        self._new_model_pipeline_file=os.path.join(self._out_folder,"pipeline.config")
        self._labels_map_file=os.path.join(self._out_folder,"label_map.pbtx")
        self._val_record_file=os.path.join(self._out_folder,"val.record")
        self._train_record_file=os.path.join(self._out_folder,"train.record")

        self._new_model_pipeline_dict=None
        self._labels_map_dict=None

    def labels(self):
        return self._dataset_df["class"].unique()

    def _load_dataset_for_detection(self):
        self._dataset_df=read_pascal_dataset(self._xml_folder)

    def _load_dataset_for_segmentation(self):
        self._dataset_df = read_pascal_dataset(self._xml_folder)

    def _mk_labels_map(self):
        if os.path.isfile(self._labels_map_file):
            os.remove(self._labels_map_file)
        with open(self._labels_map_file,'a') as f:
            for idx,name in enumerate(self.labels()):
                item="item{{\n id: {} \n name: '{}'\n}} \n".format(idx+1,name)
                f.write(item)

    @classmethod
    def _mk_record_file(cls,df: pd.DataFrame,out_file,map_labels_file):
        map_labels=get_label_map_dict(map_labels_file)
        with tf.python_io.TFRecordWriter(out_file) as writer:
            for key,rows in df.groupby(['image_path']):
                tf_example=Utilities.create_record(key,rows,map_labels)
                writer.write(tf_example.SerializeToString())

    def _mk_records(self,split_size):
        train_df,val_df=train_test_split(self._dataset_df,test_size=split_size,random_state=42,shuffle=True)
        self._mk_record_file(train_df,self._train_record_file,self._labels_map_file)
        self._mk_record_file(val_df,self._val_record_file,self._labels_map_file)

    @typeassert(epochs=int,val_split=float,override_pipeline=bool,clear_folder=bool)
    def train(self,epochs=100,val_split=0.3,clear_folder=False,override_pipeline=False):
        try:
            if clear_folder:
                FileUtil.clear_folder(self._out_folder)
            assert not self._dataset_df.empty,"the dataset is empty, or the model wasn't initialized correctly"
            self._mk_labels_map()
            self._mk_records(val_split)
            if not os.path.isfile(self._new_model_pipeline_file) or override_pipeline:
                pipeline=Utilities.load_pipeline(self._checkpoint_model_pipeline_file)
                num_classes=len(self.labels())
                pipeline=Utilities.update_pipeline(
                    pipeline,
                    num_classes,
                    str(Path(self._checkpoint_model_folder)),
                    str(Path(self._labels_map_file)),
                    str(Path(self._val_record_file)),
                    str(Path(self._train_record_file)),
                    epochs
                )
                Utilities.save_pipeline(pipeline,self._out_folder)
            os.makedirs(os.path.join(self._out_folder, "export/Servo"), exist_ok=True)
            # training
            tf.logging.set_verbosity(tf.logging.INFO)
            gpu_available = tf.test.is_gpu_available()
            if gpu_available:
                session_config=tf.ConfigProto()
                config=tf.estimator.RunConfig(
                    model_dir=self._out_folder)
            else:
                session_config=tf.ConfigProto()
                session_config.gpu_options.allow_growth=True
                config=tf.estimator.RunConfig(
                    model_dir=self._out_folder,
                    session_config=session_config)

            train_and_eval_dict=model_lib.create_estimator_and_inputs(
                run_config=config,
                hparams=model_hparams.create_hparams(None),
                pipeline_config_path=self._new_model_pipeline_file,
                train_steps=epochs,
                sample_1_of_n_eval_examples=1,
                sample_1_of_n_eval_on_train_examples=5)
            estimator=train_and_eval_dict['estimator']
            train_input_fn=train_and_eval_dict['train_input_fn']
            eval_input_fns=train_and_eval_dict['eval_input_fns']
            eval_on_train_input_fn=train_and_eval_dict['eval_on_train_input_fn']
            predict_input_fn=train_and_eval_dict['predict_input_fn']
            train_steps=train_and_eval_dict['train_steps']
            print(train_steps)
            train_spec,eval_specs=model_lib.create_train_and_eval_specs(
                train_input_fn,
                eval_input_fns,
                eval_on_train_input_fn,
                predict_input_fn,
                train_steps,
                eval_on_train_data=False)
            #print(eval_specs[0])
            # Currently only a single Eval Spec is allowed.
            tf.estimator.train_and_evaluate(estimator,train_spec,eval_specs[0])
        except Exception as ex:
            logger.error("Error training the model {}".format(ex))
        return super(TfTrainableModel,self).train()

    @typeassert(checkpoint=int,out_folder=str)
    def freeze(self,checkpoint,out_folder=None):
        try:
            tf.disable_eager_execution()
            model_checkpoint="{}/model.ckpt-{}".format(self._out_folder,checkpoint)
            if out_folder:
                frozen_model_path=os.path.join(out_folder,"frozen_inference_graph.pb")
            else:
                frozen_model_path=os.path.join(self._out_folder,"export/frozen_inference_graph.pb")
            model_pipeline=self._new_model_pipeline_file
            frozen_model_dir=os.path.dirname(frozen_model_path)
            pipeline_config=pipeline_pb2.TrainEvalPipelineConfig()
            tf.reset_default_graph()
            with tf.gfile.GFile(model_pipeline,'r') as f:
                text_format.Merge(f.read(),pipeline_config)
            exporter.export_inference_graph(
                input_type="image_tensor",
                pipeline_config=pipeline_config,
                trained_checkpoint_prefix=model_checkpoint,
                output_directory=frozen_model_dir,
                input_shape=None,
                write_inference_graph=False)
        except Exception as ex:
            logger.error("Error freezing the model {}".format(ex))
        return super(TfTrainableModel,self).freeze()

    def eval(self,*args,**kwargs):
        return super(TfTrainableModel,self).eval()

    def __enter__(self):
        try:
            self._checkpoint_model_folder=Utilities.download_model(self._model_name)
            self._checkpoint_model_pipeline_file=Utilities.download_pipeline(self._model_name)
            pipeline_dict = Utilities.load_pipeline(self._checkpoint_model_pipeline_file)
            model_config = pipeline_dict["model"]  # read Detection model
            model_arch = model_config.WhichOneof("model")
            print(model_arch)
            if model_arch in ["ssd", "faster_rcnn"]:
                self._load_dataset_for_detection()
            else:
                raise Exception("Arch not supported yet")
            return self
        except  Exception as ex:
            logger.error("Error loading the model {}".format(ex))

    def __exit__(self,exc_type,exc_val,exc_tb):
        if exc_type:
            logger.error("Error loading the model:  {}, {}".format(exc_type,str(exc_val)))
