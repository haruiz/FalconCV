import abc
import os
from abc import ABCMeta

import dask
from dask import delayed

from falconcv.util import LibUtil, FileUtil

DEP_KEY = "DEPENDENCIES"


class DatasetDownloader(metaclass=ABCMeta):
    def __init__(self):
        self._dependencies = {}
        self._images = []
        self._task = None
        self._split = None
        self._labels_map = {}
        self._slabels_map = {}
        self._remote_dep = {}

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, value):
        self._split = value

    @property
    def labels_map(self):
        return self._labels_map

    @labels_map.setter
    def labels_map(self,value):
        self._labels_map=value

    @property
    def slabels_map(self):
        return self._slabels_map

    @slabels_map.setter
    def slabels_map(self,value):
        self._slabels_map=value

    def _home(self):
        """ @:return the current dataset path """
        lib_home=LibUtil.home()
        ds_name=type(self).__name__
        ds_path=os.path.join(lib_home, "datasets", ds_name)
        if not os.path.exists(ds_path):
            os.makedirs(ds_path,exist_ok=True)
        return ds_path

    def _download_dependencies(self):
        """Download the dataset dependencies"""
        delayed_tasks= {}
        for dep_name, dep_uri in self._remote_dep.items():
            _, dep_filename = os.path.split(dep_uri)
            dep_path = os.path.join(self._home(), dep_filename)
            task = delayed(FileUtil.download_file)(dep_uri,dep_path)
            delayed_tasks[dep_name] = task
        self._dependencies =dask.compute(delayed_tasks)[0]

    def _get_dependency(self,name):
        """@:return: "a dependency path """
        return self._dependencies.get(name,None)

    @abc.abstractmethod
    def fetch(self,n=None, labels=None, batch_size : int= 200):
        raise NotImplementedError

    @abc.abstractmethod
    def setup(self, split=None, task=None):
        self._split = split
        self._task = task
        self._download_dependencies() # download the dataset files dependencies





