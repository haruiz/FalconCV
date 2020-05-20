from pathlib import Path

import tensorflow as tf
from object_detection.utils.config_util import create_pipeline_proto_from_configs, \
    save_pipeline_config


class Utilities:
    @staticmethod
    def get_files(path: Path, exts: list) -> list:
        all_files = []
        for ext in exts:
            all_files.extend(path.rglob("**/*%s" % ext))
        return all_files

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
    def save_pipeline(pipeline_dict,out_folder):
        pipeline_proto=create_pipeline_proto_from_configs(pipeline_dict)
        save_pipeline_config(pipeline_proto,out_folder)


