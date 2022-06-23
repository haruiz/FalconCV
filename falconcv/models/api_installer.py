from abc import abstractmethod, ABC


class ApiInstaller(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def install(self):
        raise NotImplementedError()
