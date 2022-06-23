from falconcv.decor import requires

try:
    import tensorflow.compat.v1 as tf
    from object_detection import (
        model_hparams,
        model_lib,
        export_tflite_ssd_graph_lib,
        exporter,
    )
    from object_detection.builders import (
        graph_rewriter_builder,
        dataset_builder,
        model_builder,
    )
    from object_detection.legacy import trainer
    from object_detection.utils.config_util import (
        create_pipeline_proto_from_configs,
        get_configs_from_pipeline_file,
        save_pipeline_config,
        create_configs_from_pipeline_proto,
        update_input_reader_config,
        _update_tf_record_input_path,
        merge_external_params_with_configs,
    )
    from object_detection.utils.label_map_util import get_label_map_dict
except ImportError as ex:
    print(ex)


@requires("tensorflow", "object_detection")
class PretrainedModelPipelineHandler:
    def __init__(self, pipeline_path):
        self._pipeline_path = pipeline_path
        self._configs = get_configs_from_pipeline_file(str(self._pipeline_path))
        self._pipeline = create_pipeline_proto_from_configs(self._configs)

    @property
    def fine_tune_checkpoint_type(self):
        return self._pipeline.train_config.fine_tune_checkpoint_type

    @fine_tune_checkpoint_type.setter
    def fine_tune_checkpoint_type(self, value):
        self._pipeline.train_config.fine_tune_checkpoint_type = value

    @property
    def use_bfloat16(self):
        return self._pipeline.train_config.use_bfloat16

    @use_bfloat16.setter
    def use_bfloat16(self, value):
        self._pipeline.train_config.use_bfloat16 = value

    @property
    def configs(self):
        return self._configs

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def arch(self):
        return self._pipeline.model.WhichOneof("model")

    @property
    def num_classes(self):
        return getattr(self._pipeline.model, self.arch).num_classes

    @num_classes.setter
    def num_classes(self, value):
        getattr(self._pipeline.model, self.arch).num_classes = value

    @property
    def input_size(self):
        resizer_config = getattr(self._pipeline.model, self.arch).image_resizer
        if resizer_config.HasField("fixed_shape_resizer"):
            sz = [
                resizer_config.fixed_shape_resizer.width,
                resizer_config.fixed_shape_resizer.height,
            ]
        elif resizer_config.HasField("keep_aspect_ratio_resizer"):
            sz = [
                resizer_config.keep_aspect_ratio_resizer.min_dimension,
                resizer_config.keep_aspect_ratio_resizer.max_dimension,
            ]
        elif resizer_config.HasField("identity_resizer") or resizer_config.HasField(
            "conditional_shape_resizer"
        ):
            sz = [-1, -1]
        else:
            raise ValueError("Unknown image resizer type.")
        return tuple(sz)

    @property
    def resizer_type(self):
        resizer_config = getattr(self._pipeline.model, self.arch).image_resizer
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
        assert isinstance(
            value, tuple
        ), "invalid input size, it must be provide in the format (300,300)"
        resizer_config = getattr(self._pipeline.model, self.arch).image_resizer
        if resizer_config.HasField("fixed_shape_resizer"):
            resizer_config.fixed_shape_resizer.width = value[0]
            resizer_config.fixed_shape_resizer.height = value[1]
        elif resizer_config.HasField("keep_aspect_ratio_resizer"):
            resizer_config.keep_aspect_ratio_resizer.min_dimension = value[0]
            resizer_config.keep_aspect_ratio_resizer.max_dimension = value[1]

    @property
    def num_steps(self):
        return self._pipeline.train_config.num_steps

    @num_steps.setter
    def num_steps(self, value):
        assert isinstance(value, int), "invalid value"
        self._pipeline.train_config.num_steps = value

    @property
    def batch_size(self):
        return self._pipeline.train_config.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._pipeline.train_config.batch_size = value

    @property
    def label_map_path(self):
        return self._pipeline.train_input_reader.label_map_path

    @label_map_path.setter
    def label_map_path(self, value):
        self._pipeline.train_input_reader.label_map_path = value

    def set_config_paths(
        self, train_record_file, val_record_file, labels_map_file, checkpoint_path
    ):
        configs = create_configs_from_pipeline_proto(self._pipeline)
        update_input_reader_config(
            configs,
            key_name="train_input_config",
            input_name=None,
            field_name="input_path",
            value=str(train_record_file),
            path_updater=_update_tf_record_input_path,
        )
        update_input_reader_config(
            configs,
            key_name="eval_input_configs",
            input_name=None,
            field_name="input_path",
            value=str(val_record_file),
            path_updater=_update_tf_record_input_path,
        )
        update_dict = {
            "label_map_path": str(labels_map_file),
            "train_config.fine_tune_checkpoint": str(checkpoint_path),
        }
        self._configs = merge_external_params_with_configs(
            configs, kwargs_dict=update_dict
        )
        self._pipeline = create_pipeline_proto_from_configs(self._configs)

    def save(self, output_folder):
        save_pipeline_config(self._pipeline, str(output_folder))
