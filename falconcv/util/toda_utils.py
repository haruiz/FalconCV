import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from matplotlib import patches

from . import FileUtils
from .vis_utils import VisUtils

logger = logging.getLogger("rich")

# JpegImagePlugin._getmp = lambda: None

try:
    import tensorflow.compat.v1 as tf
    from object_detection.utils import visualization_utils as vu, label_map_util
    from object_detection.protos import string_int_label_map_pb2 as pb
    from object_detection import exporter_lib_v2, export_tflite_graph_lib_tf2
    from object_detection.protos import pipeline_pb2
    from tensorflow_lite_support.metadata.python import metadata
    from tensorflow_lite_support.metadata.python.metadata_writers import (
        writer_utils,
        object_detector,
    )
    from object_detection.data_decoders.tf_example_decoder import (
        TfExampleDecoder as TfDecoder,
    )
    from google.protobuf import text_format
    import itertools
except ImportError:
    ...


class TODAUtils:
    @staticmethod
    def draw_mask(image_arr, mask_arr, labels_map, labels_colors, opacity):
        assert image_arr.dtype == np.uint8, "`image` not of type np.uint8"
        assert mask_arr.dtype == np.uint8, "`mask` not of type np.uint8"
        if image_arr.shape[:2] != mask_arr.shape:
            raise ValueError(
                "The image has spatial dimensions %s but the mask has "
                "dimensions %s" % (image_arr.shape[:2], mask_arr.shape)
            )
        for label in labels_map:
            if label in labels_colors:
                label_id = labels_map[label]
                color = np.asarray(labels_colors[label])
                color = color * 255
                color = color[:-1]  # not include alpha value
                # create binary mask for each label
                label_binary_mask = np.zeros_like(mask_arr)
                label_binary_mask[mask_arr == label_id] = 1
                pil_image = PILImage.fromarray(image_arr)  # output image
                solid_color = np.expand_dims(
                    np.ones_like(label_binary_mask), axis=2
                ) * np.reshape(color, [1, 1, 3])
                pil_solid_color = PILImage.fromarray(np.uint8(solid_color)).convert(
                    "RGBA"
                )
                pil_mask = PILImage.fromarray(
                    np.uint8(255.0 * opacity * mask_arr)
                ).convert("L")
                pil_image = PILImage.composite(pil_solid_color, pil_image, pil_mask)
                np.copyto(image_arr, np.array(pil_image.convert("RGB")))

    @staticmethod
    def draw_boxes(annotations, labels_colors, ax):
        for box in annotations:
            color = labels_colors[box.name.lower()]
            ax.add_patch(
                patches.Rectangle(
                    (box.xmin, box.ymin),
                    box.xmax - box.xmin,
                    box.ymax - box.ymin,
                    linewidth=3,
                    edgecolor=color[:-1],
                    fill=False,
                )
            )

    @classmethod
    def draw_labels(cls, annotations, labels_colors, ax, fontsize):
        for ann in annotations:
            color = labels_colors[ann.name.lower()]
            label = f"{ann.name} :{ann.score:.2f}"
            ax.text(
                x=ann.xmin,
                y=ann.ymin,
                s=label,
                color="white",
                fontsize=fontsize,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.9),
            )

    @staticmethod
    def export_saved_model(
        model_checkpoint_path, model_pipeline_path, out_folder, **kwargs
    ):
        config_override = kwargs.get("config_override", "")
        serve_output_folder = kwargs.get("output_folder", out_folder)
        use_side_inputs = kwargs.get("use_side_inputs", "")
        side_input_shapes = kwargs.get("side_input_shapes", "")
        side_input_types = kwargs.get("side_input_types", "")
        side_input_names = kwargs.get("side_input_names", "")
        input_type = kwargs.get("input_type", "image_tensor")
        serve_output_folder = Path(serve_output_folder)
        if serve_output_folder.exists():
            FileUtils.clear_folder(serve_output_folder)
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(model_pipeline_path, "r") as f:
            text_format.Merge(f.read(), pipeline_config)
        text_format.Merge(config_override, pipeline_config)
        exporter_lib_v2.export_inference_graph(
            input_type,
            pipeline_config,
            model_checkpoint_path,
            serve_output_folder,
            use_side_inputs,
            side_input_shapes,
            side_input_types,
            side_input_names,
        )

    @staticmethod
    def export_tf_lite(
        model_checkpoint,
        model_pipeline_path,
        model_labels_map_path,
        tflite_model_output_folder,
        **kwargs,
    ):
        # define the tflite model outputs folder
        tflite_save_model_path = tflite_model_output_folder.joinpath("saved_model")
        tflite_model_path_with_non_metadata = tflite_model_output_folder.joinpath(
            "model.tflite"
        )
        tflite_model_path_with_metadata = tflite_model_output_folder.joinpath(
            "model.metadata.tflite"
        )
        tflite_model_label_path = tflite_model_output_folder.joinpath("label_map.pbtxt")
        if tflite_model_output_folder.exists():
            FileUtils.clear_folder(tflite_model_output_folder)
        # Exports TF2 detection SavedModel for conversion to TensorFlow Lite.
        config_override = kwargs.get("config_override", "")
        max_detections = kwargs.get("max_detections", 10)
        ssd_use_regular_nms = kwargs.get("ssd_use_regular_nms", False)
        centernet_include_keypoints = kwargs.get("centernet_include_keypoints", False)
        keypoint_label_map_path = kwargs.get("keypoint_label_map_path", None)
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(model_pipeline_path, "r") as f:
            text_format.Parse(f.read(), pipeline_config)
        override_config = pipeline_pb2.TrainEvalPipelineConfig()
        text_format.Parse(config_override, override_config)
        pipeline_config.MergeFrom(override_config)
        export_tflite_graph_lib_tf2.export_tflite_model(
            pipeline_config,
            model_checkpoint,
            tflite_model_output_folder,
            max_detections,
            ssd_use_regular_nms,
            centernet_include_keypoints,
            keypoint_label_map_path,
        )
        # create tflite model
        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(tflite_save_model_path)
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_model_path_with_non_metadata, "wb") as f:
            f.write(tflite_model)
        # creates labels map
        category_index = label_map_util.create_category_index_from_labelmap(
            model_labels_map_path
        )
        arch = pipeline_config.model.WhichOneof("model")
        num_classes = getattr(pipeline_config.model, arch).num_classes
        f = open(tflite_model_label_path, "w")
        for class_id in range(1, num_classes + 1):
            if class_id not in category_index:
                f.write("???\n")
                continue
            name = category_index[class_id]["name"]
            f.write(name + "\n")
        f.close()
        # add metadata
        tflite_graph = writer_utils.load_file(str(tflite_model_path_with_non_metadata))
        writer = object_detector.MetadataWriter.create_for_inference(
            tflite_graph,
            input_norm_mean=[127.5],
            input_norm_std=[127.5],
            label_file_paths=[str(tflite_model_label_path)],
        )
        writer_utils.save_file(writer.populate(), str(tflite_model_path_with_metadata))
        displayer = metadata.MetadataDisplayer.with_model_file(
            str(tflite_model_path_with_metadata)
        )
        logger.info("Metadata populated:")
        logger.info(displayer.get_metadata_json())
        logger.info("=============================")
        logger.info("Associated file(s) populated:")
        logger.info(displayer.get_packed_associated_file_list())

    @classmethod
    def visualize_tf_record(
        cls, tfrecords_file, label_map_file, matplotlib_backend="tkagg"
    ):
        with VisUtils.with_matplotlib_backend(matplotlib_backend):
            if label_map_file is not None:
                label_map_proto = pb.StringIntLabelMap()
                with tf.gfile.GFile(label_map_file, "r") as f:
                    text_format.Merge(f.read(), label_map_proto)
                    class_dict = {}
                    for entry in label_map_proto.item:
                        class_dict[entry.id] = {"name": entry.name}
            with tf.compat.v1.Session() as sess:
                decoder = TfDecoder(
                    label_map_proto_file=label_map_file, use_display_name=False
                )
                sess.run(tf.tables_initializer())
                topN = itertools.islice(
                    tf.python_io.tf_record_iterator(tfrecords_file), 5
                )
                for record in topN:
                    example = decoder.decode(record)
                    host_example = sess.run(example)
                    scores = np.ones(host_example["groundtruth_boxes"].shape[0])
                    vu.visualize_boxes_and_labels_on_image_array(
                        host_example["image"],
                        host_example["groundtruth_boxes"],
                        host_example["groundtruth_classes"],
                        scores,
                        class_dict,
                        max_boxes_to_draw=None,
                        use_normalized_coordinates=True,
                    )
                    plt.imshow(host_example["image"])
                    plt.show()

    @classmethod
    def load_label_map(cls, label_map_path):
        from object_detection.utils.label_map_util import (
            create_category_index_from_labelmap,
        )

        labels_map = create_category_index_from_labelmap(label_map_path)
        labels_map = {v["name"]: k for k, v in labels_map.items()}
        return labels_map
