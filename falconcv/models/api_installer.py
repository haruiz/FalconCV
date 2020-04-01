import logging
import os
from abc import abstractmethod, ABCMeta
from pathlib import Path

import git
import validators
from clint.textui import progress
from git import Repo, RemoteProgress

from falconcv.util import LibUtil

logger = logging.getLogger(__name__)


class CloneProgress(RemoteProgress):
    def __init__(self, repo_name):
        super(CloneProgress, self).__init__()
        self.bar = None
        self.repo_name = repo_name

    def update(self, op_code, cur_count, max_count=None, message=''):
        is_begin = op_code & git.RemoteProgress.BEGIN != 0
        # is_end=op_code & git.RemoteProgress.END != 0
        if is_begin:
            if self.bar is None:
                self.bar = progress.Bar(
                    label="cloning {} repository".format(self.repo_name),
                    expected_size=max_count)
        else:
            if op_code != cur_count:
                self.bar.show(cur_count)  # update progress bar


class ApiInstaller(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self._package_name = None
        self._repo_uri = None
        self._repo_folder = None

    @property
    def repo_uri(self) -> str:
        return self._repo_uri

    @repo_uri.setter
    def repo_uri(self, value):
        self._repo_uri = value

    @property
    def repo_folder(self) -> Path:
        return self._repo_folder

    @repo_folder.setter
    def repo_folder(self, value):
        self._repo_folder = value

    @staticmethod
    def get_repo_name_from_url(url: str) -> str:
        last_slash_index = url.rfind("/")
        last_suffix_index = url.rfind(".git")
        if last_suffix_index < 0:
            last_suffix_index = len(url)
        if last_slash_index < 0 or last_suffix_index <= last_slash_index:
            raise Exception("invalid url format {}".format(url))
        return url[last_slash_index + 1:last_suffix_index]

    def _clone_repo(self):
        try:
            assert self._repo_uri and validators.url(self._repo_uri), "Invalid repo uri"
            repo_name = self.get_repo_name_from_url(self._repo_uri)
            self._repo_folder = LibUtil.repos_home().joinpath(repo_name)
            self._repo_folder.mkdir(exist_ok=True)
            if not self._repo_folder.joinpath(".git").is_dir():
                Repo.clone_from(self._repo_uri,
                                self._repo_folder,
                                depth=1,
                                branch='master',
                                progress=CloneProgress(repo_name))
        except git.exc.InvalidGitRepositoryError as ex:
            logger.error("Error cloning the repository from {} : {}".format(self._repo_uri, ex))
        except git.exc.GitError as ex:
            logger.error("Error cloning the repository from {} : {}".format(self._repo_uri, ex))

    @abstractmethod
    def install(self):
        self._clone_repo()
