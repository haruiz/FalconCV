import io
import os
from pathlib import Path
from object_detection.utils import dataset_util
from object_detection.utils.dataset_util import  *
import tensorflow as tf
from PIL import Image
from object_detection.utils.config_util import get_configs_from_pipeline_file,create_pipeline_proto_from_configs, \
    save_pipeline_config,update_input_reader_config,_update_tf_record_input_path,merge_external_params_with_configs
from lxml import etree
import hashlib
import numpy as np


class Utilities:
    @staticmethod
    def get_files(path: Path, exts: list) -> list:
        all_files = []
        for ext in exts:
            all_files.extend(path.rglob("**/*%s" % ext))
        return all_files

    @staticmethod
    def image_to_example(img_path: Path, xml_path: Path, mask_path: Path, labels_map=None):
        # load annotations
        annotations = None
        if xml_path.exists():
            with tf.io.gfile.GFile(str(xml_path), 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            annotations = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        # read image
        try:
            with tf.io.gfile.GFile(str(img_path), 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image: Image = Image.open(encoded_jpg_io)
            width, height = image.size
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            image_key = hashlib.sha256(encoded_jpg).hexdigest()
        except Exception as ex:
            print("Error encoding image {}, image ignored".format(img_path))
            return None

        # read mask
        mask = None
        if mask_path and mask_path.exists():
            with tf.io.gfile.GFile(str(mask_path), 'rb') as fid:
                encoded_mask_png = fid.read()
            encoded_png_io = io.BytesIO(encoded_mask_png)
            mask: Image = Image.open(encoded_png_io)
            width, height = mask.size
            if mask.format != 'PNG':
                raise ValueError('Image format not PNG')
            mask = np.asarray(mask)

        # create records
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        classes, classes_text, encoded_masks = [], [], []
        encoded_mask_png_list = []
        if annotations:
            if 'object' in annotations:
                for obj in annotations['object']:
                    class_name = obj['name'].title()
                    class_id = labels_map[class_name]
                    xmin = float(obj['bndbox']['xmin'])
                    xmax = float(obj['bndbox']['xmax'])
                    ymin = float(obj['bndbox']['ymin'])
                    ymax = float(obj['bndbox']['ymax'])
                    xmins.append(xmin / width)
                    ymins.append(ymin / height)
                    xmaxs.append(xmax / width)
                    ymaxs.append(ymax / height)
                    classes_text.append(class_name.encode('utf8'))
                    classes.append(class_id)
                    # if a mask exist
                    if isinstance(mask, np.ndarray):
                         # object mask
                        mask_roi = np.zeros_like(mask)
                        mask_roi[int(ymin):int(ymax), int(xmin):int(xmax)] = mask[int(ymin):int(ymax),
                                                                             int(xmin):int(xmax)]
                        mask_remapped = (mask_roi == class_id).astype(np.uint8)
                        mask_remapped = Image.fromarray(mask_remapped)
                        output = io.BytesIO()
                        mask_remapped.save(output, format='PNG')
                        encoded_mask_png_list.append(output.getvalue())

        feature_dict = {
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/filename': bytes_feature(img_path.name.encode('utf8')),
            'image/source_id': bytes_feature(img_path.name.encode('utf8')),
            'image/key/sha256': bytes_feature(image_key.encode('utf8')),
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmins),
            'image/object/bbox/xmax': float_list_feature(xmaxs),
            'image/object/bbox/ymin': float_list_feature(ymins),
            'image/object/bbox/ymax': float_list_feature(ymaxs),
            'image/object/class/text': bytes_list_feature(classes_text),
            'image/object/class/label': int64_list_feature(classes)}
        if len(encoded_mask_png_list) > 0:
            feature_dict['image/object/mask'] = bytes_list_feature(encoded_mask_png_list)
        tf_data = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return tf_data

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
    def update_pipeline(pipeline_dict,num_classes,checkpoint_model_folder,labels_map_file,val_record_f,train_record_f,  epochs):
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
