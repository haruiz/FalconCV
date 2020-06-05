import glob
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

from falconcv.models import ApiInstaller
from falconcv.util import FileUtil

logger = logging.getLogger(__name__)


class TFObjectDetectionAPI(ApiInstaller):
    def __init__(self):
        super(TFObjectDetectionAPI, self).__init__()
        self.repo_uri = "https://github.com/tensorflow/models.git"
        self._package_name = "object_detection"


    def install(self):
        try:
            super(TFObjectDetectionAPI, self).install()
            self._protobuf_comp()
            research_folder = self.repo_folder.joinpath("research")
            slim_folder = research_folder.joinpath("slim")
            if importlib.util.find_spec(self._package_name) is None:
                logger.debug("Installing Api")
                with FileUtil.workon(str(research_folder)):
                    os.system("python setup.py build")
                    os.system("python setup.py install")
                logger.debug("Api installation done")
            sys.path.append(str(research_folder))
            sys.path.append(str(slim_folder))
            os.environ['PATH'] += "{}{}{}".format(str(research_folder), os.pathsep, str(slim_folder))
        except Exception as ex:
            logger.error("Error installing the package : {}".format(ex))

    def _protobuf_comp(self):
        research_folder = self.repo_folder.joinpath("research")
        protos_folder = research_folder.joinpath("object_detection", "protos")
        protos_files = glob.glob("{}/*.proto".format(str(protos_folder)))
        for abs_file_path in protos_files:
            try:
                file_name = Path(abs_file_path).name
                rel_file_path = "object_detection/protos/{}".format(file_name)
                p = subprocess.Popen(
                    ['protoc', rel_file_path, "--python_out=."],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    cwd=str(research_folder))
                output = p.stdout.readlines()
                error = p.stderr.readlines()
                if error:
                    raise IOError(error)
                if output:
                    print(output)
                    logger.debug(output)
                p.wait()
            except Exception as ex:
                print(ex)
                continue


if __name__ == '__main__':
    api = TFObjectDetectionAPI()
    api.install()
