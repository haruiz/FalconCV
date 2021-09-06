import tensorflow as tf
import pkg_resources
from object_detection.utils.config_util import create_pipeline_proto_from_configs, \
    save_pipeline_config
import numpy as np
import itertools
from object_detection.inference import detection_inference
tf.logging.set_verbosity(tf.logging.INFO)
import pandas as pd

class Utilities:
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
    def generate_detections_record(input_tfrecord_paths, output_tfrecord_path, inference_graph):
        with tf.Session() as sess:
            input_tfrecord_paths = [
                v for v in input_tfrecord_paths.split(',') if v]
            tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
            serialized_example_tensor, image_tensor = detection_inference.build_input(
                input_tfrecord_paths)
            tf.logging.info('Reading graph and building model...')
            (detected_boxes_tensor, detected_scores_tensor,
             detected_labels_tensor) = detection_inference.build_inference_graph(
                image_tensor, inference_graph)

            tf.logging.info('Running inference and writing output to {}'.format(
                output_tfrecord_path))
            sess.run(tf.local_variables_initializer())
            tf.train.start_queue_runners()
            with tf.python_io.TFRecordWriter(output_tfrecord_path) as tf_record_writer:
                try:
                    for counter in itertools.count():
                        tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10, counter)
                        tf_example = detection_inference.infer_detections_and_add_to_example(
                            serialized_example_tensor, detected_boxes_tensor,
                            detected_scores_tensor, detected_labels_tensor,
                            False)
                        tf_record_writer.write(tf_example.SerializeToString())
                except tf.errors.OutOfRangeError:
                    tf.logging.info('Finished processing records')

    @staticmethod
    def confusion_matrix_to_dataframe(confusion_matrix, categories, iou_threshold):
        print("\nConfusion Matrix:")
        print(confusion_matrix, "\n")
        results = []

        for i in range(len(categories)):
            id = categories[i]["id"] - 1
            name = categories[i]["name"]

            total_target = np.sum(confusion_matrix[id, :])
            total_predicted = np.sum(confusion_matrix[:, id])

            precision = float(confusion_matrix[id, id] / total_predicted)
            recall = float(confusion_matrix[id, id] / total_target)
            # print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
            # print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))
            results.append({'category': name, 'precision_@{}IOU'.format(iou_threshold): precision,
                            'recall_@{}IOU'.format(iou_threshold): recall})

        return pd.DataFrame(results)

    @staticmethod
    def compute_iou(groundtruth_box, detection_box):
        g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
        d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())

        xa = max(g_xmin, d_xmin)
        ya = max(g_ymin, d_ymin)
        xb = min(g_xmax, d_xmax)
        yb = min(g_ymax, d_ymax)

        intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

        boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
        boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

        return intersection / float(boxAArea + boxBArea - intersection)

    @staticmethod
    def save_pipeline(pipeline_dict,out_folder):
        pipeline_proto=create_pipeline_proto_from_configs(pipeline_dict)
        save_pipeline_config(pipeline_proto,out_folder)

    @staticmethod
    def get_openvino_front_file(arch, pre_trained=False):
        tf_version = pkg_resources.get_distribution("tensorflow").version
        tf_version = tf_version[:-2]
        supported_arch = ["ssd", "faster", "mask", "rfcn"]
        supported_api_version = ["1.14", "1.15"]
        assert tf_version in supported_api_version, "tf version not supported"
        assert arch in supported_arch, "model arch not supported"
        if pre_trained:
            front_files_map ={
                "ssd": "ssd_v2_support.json",
                "faster": "faster_rcnn_support.json",
                "mask": "mask_rcnn_support.json",
                "rfcn": "rfcn_support.json "
            }
            front_file= front_files_map[arch]
        else:
            if arch == "ssd":
                front_file= "ssd_support_api_v{}.json".format(tf_version)
            elif arch == "faster":
                front_file= "{}_rcnn_support_api_v{}.json".format(arch, tf_version)
            else:
                front_file= "rfcn_support_api_v{}.json".format(tf_version)
        return front_file





