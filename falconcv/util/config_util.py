from omegaconf import OmegaConf


def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class ConfigUtil(object):
    def __init__(self):
        self.config = OmegaConf.load("config.yml")

    def get_detectron_model_zoo_url(self):
        return self.config.detectron.model_zoo_url
