from .api_installer import TFObjectDetectionAPI
from .tf_model_zoo import TFModelZoo
from .model_factory import APIModelFactory

api = TFObjectDetectionAPI()
api.install()
