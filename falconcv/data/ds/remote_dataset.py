import logging
from abc import abstractmethod
from pathlib import Path

from falconcv.data.ds.dataset import Dataset
from falconcv.util import LibUtil, FileUtil

logger = logging.getLogger(__name__)


class RemoteDataset(Dataset):
    @abstractmethod
    def __init__(self, version: int = 2017, split: str = "train", labels: [str] = None, n_images: int = 0, batch_size: int = 12):
        super().__init__(batch_size)
        self._split = split
        self._version = version
        self._n_images = n_images
        self._labels = labels
        self._available_labels = {}

    @property
    def available_labels(self):
        return self._available_labels

    def home(self) -> Path:
        """ @:return the current dataset home """
        ds_home = LibUtil.datasets_home()
        ds_name = type(self).__name__
        ds_path = ds_home.joinpath(ds_name)
        ds_path.mkdir(exist_ok=True)
        return ds_path

    def _download_dependencies(self):
        """Download the dataset dependencies"""
        print("Downloading {} dataset dependencies, it can take a few minutes".format(type(self).__name__))
        for dep_name, dep_uri in self._files.items():
            self._deps[dep_name] = FileUtil.download_file(dep_uri, self.home(), show_progress=True, unzip=True)
        print("Download dependencies done")

    def _get_dependency(self, name):
        """
        get a dependency path by name
        @:return: "a dependency path """
        return str(self._deps.get(name, None))

    def __load__(self):
        self._download_dependencies()  # download remote dependencies
