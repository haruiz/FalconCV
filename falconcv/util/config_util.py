import os


class ConfigUtil(object):
    @staticmethod
    def get_detectron_model_zoo_url() -> str:
        return os.environ["DETECTRON2_MODEL_ZOO_URI"]
