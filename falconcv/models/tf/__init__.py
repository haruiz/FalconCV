from .api_installer import TFObjectDetectionAPI
api = TFObjectDetectionAPI()
api.install()
from .zoo import ModelZoo
from .model_factory import APIModelFactory
