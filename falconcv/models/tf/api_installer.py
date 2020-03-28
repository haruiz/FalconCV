import glob
import os
import subprocess
from falconcv.models import ApiInstaller
import importlib.util
import logging
import sys
from subprocess import call
logger=logging.getLogger(__name__)



class  TFObjectDetectionAPI(ApiInstaller):
    def __init__(self):
        super(TFObjectDetectionAPI, self).__init__()
        self._repo_uri = "https://github.com/tensorflow/models.git"
        self._package_name = "object_detection"

    def install(self):
        try:
            super(TFObjectDetectionAPI, self).install()
            self._protobuf_comp()
            if importlib.util.find_spec(self._package_name) is None:
                call("python setup.py build", cwd=os.path.join(self.repo_folder, "research"))
                call("python setup.py install", cwd=os.path.join(self.repo_folder, "research"))
            research_folder_path = os.path.sep.join([self._repo_folder,"research"])
            slim_folder_path = os.path.sep.join([self._repo_folder,"research","slim"])
            sys.path.append(research_folder_path)
            sys.path.append(slim_folder_path)
            os.environ['PYTHONPATH']+="{}:{}".format(research_folder_path, slim_folder_path)
        except Exception as ex:
            logger.error("Error installing the package : {}".format(ex))

    def _protobuf_comp(self):
        cwd=os.path.join(self._repo_folder,"research")
        protos_folder=os.path.sep.join(
            [self._repo_folder,
             "research",
             "object_detection",
             "protos"])
        protos_folder=os.path.realpath(protos_folder)
        protos_files=glob.glob("{}/*.proto".format(protos_folder))
        for file_path in protos_files:
            _,filename=os.path.split(file_path)
            p=subprocess.Popen(
                ['protoc',"object_detection/protos/{}".format(filename),"--python_out=."],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,cwd=cwd)
            output=p.stdout.readlines()
            error=p.stderr.readlines()
            if error:
                print(error)
            if output:
                print(output)
            p.wait()


