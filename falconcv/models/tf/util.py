import io
import os

import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils.config_util import get_configs_from_pipeline_file,create_pipeline_proto_from_configs, \
    save_pipeline_config,update_input_reader_config,_update_tf_record_input_path,merge_external_params_with_configs


class Utilities:

    @staticmethod
    def create_record(image_path: str,instances: pd.DataFrame,labels_map: dict):
        from object_detection.utils.dataset_util import (bytes_feature,
                                                         bytes_list_feature,
                                                         float_list_feature,
                                                         int64_feature,
                                                         int64_list_feature)
        image_name=os.path.basename(image_path).encode("utf8")
        image_ext=os.path.splitext(image_path)[1].lower().encode("utf8")
        fid=tf.gfile.GFile(image_path,'rb')
        image_encoded=fid.read()
        fid.close()
        image_buffer=io.BytesIO(image_encoded)
        image_raw=Image.open(image_buffer)
        image_width=image_raw.size[0]
        image_height=image_raw.size[1]
        xmins=[]
        xmaxs=[]
        ymins=[]
        ymaxs=[]
        classes=[]
        classes_names=[]
        for _,entry in instances.iterrows():
            xmins.append(entry['xmin']/image_width)
            xmaxs.append(entry['xmax']/image_width)
            ymins.append(entry['ymin']/image_height)
            ymaxs.append(entry['ymax']/image_height)
            class_name=entry['class'].encode('utf8')
            classes_names.append(class_name)
            class_id=labels_map[entry['class']]
            classes.append(class_id)

        tf_example=tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height': int64_feature(image_height),
                    'image/width': int64_feature(image_width),
                    'image/filename': bytes_feature(image_name),
                    'image/source_id': bytes_feature(image_name),
                    'image/encoded': bytes_feature(image_encoded),
                    'image/format': bytes_feature(image_ext),
                    'image/object/bbox/xmin': float_list_feature(xmins),
                    'image/object/bbox/xmax': float_list_feature(xmaxs),
                    'image/object/bbox/ymin': float_list_feature(ymins),
                    'image/object/bbox/ymax': float_list_feature(ymaxs),
                    'image/object/class/text': bytes_list_feature(classes_names),
                    'image/object/class/label': int64_list_feature(classes),
                }))
        return tf_example

    @staticmethod
    def load_save_model(model_path):
        model=tf.compat.v2.saved_model.load(model_path,None)
        return model.signatures['serving_default']

    @staticmethod
    def load_graph(model_path):
        graph=tf.compat.v1.Graph()
        with graph.as_default():
            graph_def=tf.compat.v1.GraphDef()
            with tf.gfile.GFile(model_path,"rb") as f:
                graph_def.ParseFromString(f.read())
                tf.compat.v1.import_graph_def(graph_def,name="")
        return graph,tf.compat.v1.Session(graph=graph)

    @staticmethod
    def load_pipeline(pipeline_file):
        return get_configs_from_pipeline_file(pipeline_file)

    @staticmethod
    def save_pipeline(pipeline_dict,out_folder):
        pipeline_proto=create_pipeline_proto_from_configs(pipeline_dict)
        save_pipeline_config(pipeline_proto,out_folder)

    @staticmethod
    def update_pipeline(pipeline_dict,num_classes,checkpoint_model_folder,labels_map_file,val_record_f,train_record_f,
                        epochs):
        update_input_reader_config(pipeline_dict,
                                   key_name="train_input_config",
                                   input_name=None,
                                   field_name="input_path",
                                   value=val_record_f,
                                   path_updater=_update_tf_record_input_path)

        update_input_reader_config(pipeline_dict,
                                   key_name="eval_input_configs",
                                   input_name=None,
                                   field_name="input_path",
                                   value=train_record_f,
                                   path_updater=_update_tf_record_input_path)

        checkpoint_folder="{}/model.ckpt".format(checkpoint_model_folder)
        model_config=pipeline_dict["model"]  # read Detection model
        model_arch=model_config.WhichOneof("model")
        return merge_external_params_with_configs(
            pipeline_dict,
            kwargs_dict={
                "label_map_path": labels_map_file,
                "train_config.fine_tune_checkpoint": os.path.abspath(checkpoint_folder),
                "train_config.num_steps": epochs,
                "model.{}.num_classes".format(model_arch): num_classes
            })
