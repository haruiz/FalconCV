import os


class ConfigUtil(object):
    @staticmethod
    def get_tfodapi_model_zoo_uri() -> str:
        return os.environ["TF_OBJECT_DETECTION_MODEL_ZOO_URI"]

    @staticmethod
    def get_tfodapi_model_config_uri() -> str:
        return os.environ["TF_OBJECT_DETECTION_MODEL_CONFIG_URI"]
