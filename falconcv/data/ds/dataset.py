import math
from abc import abstractmethod, ABCMeta
from pathlib import Path

import more_itertools

from falconcv.util import LibUtil


class Dataset(metaclass=ABCMeta):
    def __init__(self, batch_size=1):
        self._images: list = []
        self._task: str = ""
        self._current_batch = None
        self._current_batch_idx = 0
        self._batch_size = batch_size
        self._batches = None  # list of batches
        self._deps = {}
        self._files = {}

    @property
    def images(self) -> []:
        return self._images

    @images.setter
    def images(self, value):
        self._images = value

    def home(self) -> Path:
        """ @:return the current dataset home """
        ds_home = LibUtil.datasets_home()
        ds_name = type(self).__name__
        ds_path = ds_home.joinpath(ds_name)
        ds_path.mkdir(exist_ok=True)
        return ds_path

    @property
    def batches(self) -> list:
        """
        Create image batches
        :rtype: object
        """
        if self._batches is None:
            self._batches = list(more_itertools.chunked(self._images, self._batch_size))
        return self._batches

    @property
    def batches_count(self) -> int:
        """
        return the number of batches
        :rtype: object
        """
        return math.ceil(len(self._images) / self._batch_size)

    def __getitem__(self, item):
        return item

    def __getbatch__(self, batch):
        return batch

    @abstractmethod
    def __load__(self):
        raise NotImplementedError

    def __next__(self):
        if self._current_batch_idx >= self.batches_count:
            raise StopIteration()
        self._current_batch = self.__getbatch__(self.batches[self._current_batch_idx])
        self._current_batch_idx += 1
        return list(map(lambda item: self.__getitem__(item), self._current_batch))

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._images)
