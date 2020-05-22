from .api_installer import TFObjectDetectionAPI
from .zoo import ModelZoo
from .model_factory import APIModelFactory

api = TFObjectDetectionAPI()
api.install()
