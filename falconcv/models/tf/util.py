import tensorflow as tf
import pkg_resources
from object_detection.utils.config_util import create_pipeline_proto_from_configs, \
    save_pipeline_config


class Utilities:
    @staticmethod
    def load_save_model(model_path):
        model = tf.compat.v2.saved_model.load(model_path, None)
        return model.signatures['serving_default']

    @staticmethod
    def load_graph(model_path):
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.gfile.GFile(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.compat.v1.import_graph_def(graph_def, name="")
        return graph, tf.compat.v1.Session(graph=graph)

    @staticmethod
    def save_pipeline(pipeline_dict, out_folder):
        pipeline_proto = create_pipeline_proto_from_configs(pipeline_dict)
        save_pipeline_config(pipeline_proto, out_folder)

    @staticmethod
    def get_openvino_front_file(arch, pre_trained=False):
        tf_version = pkg_resources.get_distribution("tensorflow").version
        tf_version = tf_version[:-2]
        supported_arch = ["ssd", "faster", "mask", "rfcn"]
        supported_api_version = ["1.14", "1.15"]
        assert tf_version in supported_api_version, "tf version not supported"
        assert arch in supported_arch, "model arch not supported"
        if pre_trained:
            front_files_map = {
                "ssd": "ssd_v2_support.json",
                "faster": "faster_rcnn_support.json",
                "mask": "mask_rcnn_support.json",
                "rfcn": "rfcn_support.json "
            }
            front_file = front_files_map[arch]
        else:
            if arch == "ssd":
                front_file = "ssd_support_api_v{}.json".format(tf_version)
            elif arch == "faster":
                front_file = "{}_rcnn_support_api_v{}.json".format(arch, tf_version)
            else:
                front_file = "rfcn_support_api_v{}.json".format(tf_version)
        return front_file
