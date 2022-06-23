from abc import ABCMeta


class ApiModel(metaclass=ABCMeta):
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()
