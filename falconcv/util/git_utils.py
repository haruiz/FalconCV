import logging
import typing
from pathlib import Path

import git
import validators
from alive_progress import alive_bar
from git import Repo

logger = logging.getLogger("rich")


class GitRemoteProgress(git.RemoteProgress):
    OP_CODES = [
        "BEGIN",
        "CHECKING_OUT",
        "COMPRESSING",
        "COUNTING",
        "END",
        "FINDING_SOURCES",
        "RECEIVING",
        "RESOLVING",
        "WRITING",
    ]
    OP_CODE_MAP = {
        getattr(git.RemoteProgress, _op_code): _op_code for _op_code in OP_CODES
    }

    def __init__(self) -> None:
        super().__init__()
        self.alive_bar_instance = None

    @classmethod
    def get_curr_op(cls, op_code: int) -> str:
        """Get OP name from OP code."""
        # Remove BEGIN- and END-flag and get op name
        op_code_masked = op_code & cls.OP_MASK
        return cls.OP_CODE_MAP.get(op_code_masked, "?").title()

    def update(
        self,
        op_code: int,
        cur_count,
        max_count=None,
        message="",
    ) -> None:
        # Start new bar on each BEGIN-flag
        if op_code & self.BEGIN:
            self.curr_op = self.get_curr_op(op_code)
            self._dispatch_bar(title=self.curr_op)

        self.bar(cur_count / max_count)
        self.bar.text(message)

        # End progress monitoring on each END-flag
        if op_code & git.RemoteProgress.END:
            self._destroy_bar()

    def _dispatch_bar(self, title="") -> None:
        """Create a new progress bar"""
        self.alive_bar_instance = alive_bar(manual=True, title=title)
        self.bar = self.alive_bar_instance.__enter__()

    def _destroy_bar(self) -> None:
        """Destroy an existing progress bar"""
        self.alive_bar_instance.__exit__(None, None, None)


class GitUtils:
    @staticmethod
    def is_git_repo(path):
        """
        Check if a given folder is a git repository
        :param path:
        :return: True if the given folder is a repor or false otherwise
        """
        try:
            _ = git.Repo(path).git_dir
            return True
        except (git.exc.InvalidGitRepositoryError, Exception):
            return False

    @staticmethod
    def get_repo_name(url: str) -> str:
        """
        Get and return the repo name from a valid github url
        :rtype: str
        """
        last_slash_index = url.rfind("/")
        last_suffix_index = url.rfind(".git")
        if last_suffix_index < 0:
            last_suffix_index = len(url)
        if last_slash_index < 0 or last_suffix_index <= last_slash_index:
            raise Exception("invalid url format {}".format(url))
        return url[last_slash_index + 1 : last_suffix_index]

    @classmethod
    def clone_repo(
        cls,
        github_repo_uri: str,
        output_folder: typing.Union[str, Path],
        branch_name: str = "master",
    ) -> Repo:
        """
        Download a folder repository into the specified output folder
        :param github_repo_uri:  github repo url
        :param output_folder:  folder wher the repo will be cloned
        :param branch_name:  github repo branch that will be cloned
        :return:  cloned repo
        """
        assert github_repo_uri and validators.url(github_repo_uri), "Invalid github uri"
        try:
            repo_name = cls.get_repo_name(github_repo_uri)
            output_folder = Path(output_folder)
            output_folder = output_folder.joinpath(repo_name)
            output_folder.mkdir(parents=True, exist_ok=True)
            if cls.is_git_repo(output_folder):
                return Repo(output_folder)
            else:
                logger.info(
                    f"Cloning {repo_name} repository from {github_repo_uri} into {output_folder}...."
                )
                cloned_repo = Repo.clone_from(
                    url=github_repo_uri,
                    to_path=output_folder,
                    depth=1,
                    branch=branch_name,
                    progress=GitRemoteProgress(),
                )
            return cloned_repo
        except (git.exc.InvalidGitRepositoryError, git.exc.GitError) as ex:
            raise Exception(f"Error cloning the repository  {github_repo_uri}: {ex}")
