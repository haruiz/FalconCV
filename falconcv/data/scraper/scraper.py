import abc


class ImagesScraper(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fetch(self, *args, **kwargs):
        raise NotImplementedError
