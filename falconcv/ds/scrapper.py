import abc


class ImagesScrapper(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fetch(self,*args,**kwargs):
        raise NotImplementedError




