from abc import abstractmethod, ABCMeta

import validators
from git import Repo, RemoteProgress
import importlib
import git
from clint.textui import progress

import os
import pkg_resources  # part of setuptools

from falconcv.util import LibUtil
import logging
logger=logging.getLogger(__name__)


class CloneProgress(RemoteProgress):
    def __init__(self):
        super(CloneProgress, self).__init__()
        self.bar = None

    def update(self, op_code, cur_count, max_count=None, message=''):
        is_begin=op_code & git.RemoteProgress.BEGIN != 0
        is_end=op_code & git.RemoteProgress.END != 0
        if is_begin:
            if self.bar is None:
                self.bar = progress.Bar(
                    label="cloning repository",
                    expected_size=max_count)
        else:
            if op_code != cur_count:
                self.bar.show(cur_count) # update progress bar


class ApiInstaller(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self._package_name = None
        self._repo_uri=None
        self._repo_folder = None

    @staticmethod
    def get_repo_name_from_url(url: str) -> str:
        last_slash_index=url.rfind("/")
        last_suffix_index=url.rfind(".git")
        if last_suffix_index < 0:
            last_suffix_index=len(url)
        if last_slash_index < 0 or last_suffix_index <= last_slash_index:
            raise Exception("Badly formatted url {}".format(url))
        return url[last_slash_index+1:last_suffix_index]

    def _clone_repo(self):
        try:
            assert self._repo_uri and validators.url(self._repo_uri), "Invalid repo uri"
            repo_name = self.get_repo_name_from_url(self._repo_uri)
            self._repo_folder = os.path.sep.join([LibUtil.home(),"repos" ,repo_name])
            os.makedirs(self._repo_folder,exist_ok=True)
            if not os.path.isdir(os.path.join(self._repo_folder, ".git")):
                Repo.clone_from(self._repo_uri,
                    self._repo_folder,
                    depth=1,
                    branch='master',
                    progress=CloneProgress())
        except git.exc.InvalidGitRepositoryError as ex:
            logger.error("Error cloning the repository from {} : {}".format(self._repo_uri,ex))
        except git.exc.GitError as ex:
            logger.error("Error cloning the repository from {} : {}".format(self._repo_uri, ex))

    @abstractmethod
    def install(self):
        self._clone_repo()

