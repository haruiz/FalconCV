import importlib.util
import logging
import os
import subprocess
from pathlib import Path

from falconcv.decor import exception, cmd_depends
from falconcv.models.api_installer import ApiInstaller
from falconcv.util import GitUtils, LibUtils, FileUtils

logger = logging.getLogger("rich")


class TODAInstaller(ApiInstaller):
    def __init__(self):
        super(TODAInstaller, self).__init__()
        self.repo_uri = "https://github.com/tensorflow/models"

    @exception
    def install(self):
        if importlib.util.find_spec("object_detection"):
            return
        logger.info("Installing Tensorflow Object Detection API...")
        self._clone_models_repo()
        logging.info("Compiling protobuf files...")
        self._compile_protobuf_files()
        logger.info("running setup.py file")
        self._setup()

    def _clone_models_repo(self):
        GitUtils.clone_repo(self.repo_uri, LibUtils.repos_home())

    @staticmethod
    def _get_research_folder_path():
        models_repo_folder = LibUtils.repos_home().joinpath("models")
        return models_repo_folder.joinpath("research")

    @classmethod
    def _get_setup_file(cls):
        object_detection_folder = cls._get_research_folder_path().joinpath(
            "object_detection"
        )
        return object_detection_folder.joinpath("packages/tf2/setup.py")

    @classmethod
    @exception
    @cmd_depends("protoc")
    def _compile_protobuf_files(cls):

        research_folder = cls._get_research_folder_path()
        protos_files = research_folder.glob(f"object_detection/protos/*.proto")
        for protocf_abs_path in protos_files:
            protocf_rel_path = protocf_abs_path.relative_to(research_folder)
            p = subprocess.Popen(
                ["protoc", str(protocf_rel_path), "--python_out=."],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=str(research_folder),
            )
            output = p.stdout.readlines()
            error = p.stderr.readlines()
            if error:
                print(error)
                raise IOError(error)
            if output:
                logging.debug(output)
            p.wait()

    @classmethod
    @exception
    def _setup(self):
        research_folder = self._get_research_folder_path()
        setup_file = self._get_setup_file()
        with FileUtils.workon(research_folder):
            FileUtils.cp(setup_file, Path.cwd())
            os.system("python -m pip install .")
        try:
            import object_detection
        except ImportError:
            raise Exception("error trying to  import object_detection")


if __name__ == "__main__":
    TODAInstaller().install()
