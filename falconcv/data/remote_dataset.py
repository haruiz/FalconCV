from abc import ABCMeta

from falconcv.util import (
    LibUtils,
    FileUtils,
)


class RemoteDataset(metaclass=ABCMeta):
    def __init__(self):
        self._images = []
        self._deps_map = {}

    def home(self):
        """@:return the current dataset home"""
        return LibUtils.datasets_home(type(self).__name__)

    def _get_dependency(self, name):
        return self._deps_map[name]

    def _download_dependencies(self):
        for name, url in self._deps_map.items():
            self._deps_map[name] = FileUtils.get_file_from_uri(
                url, self.home(), show_progress=True
            )

    def __len__(self):
        return len(self._images)
