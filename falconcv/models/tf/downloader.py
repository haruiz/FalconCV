import os
import tarfile

import validators
from falconcv.util import FileUtil,Console
import logging
logger=logging.getLogger(__name__)


class ModelDownloader:

    @classmethod
    def download_od_api_model(cls,model_uri: str,out_folder: str) -> dict:
        try:
            assert validators.url(model_uri),"invalid model uri param"
            assert FileUtil.exists_http_file(model_uri),"The file doesn't exist"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)
                logger.info("Downloading model from zoo...")
                _,filename=os.path.split(model_uri)
                model_file=os.path.join(out_folder,filename)
                model_file=FileUtil.download_file(model_uri,model_file)
                with tarfile.open(model_file) as tar:
                    for member in tar.getmembers():
                        member: tarfile.TarInfo
                        path = member.name.split("/")
                        if len(path) > 0:
                            if member.isdir():
                                dir_path = os.path.join(out_folder, os.path.sep.join(path[1:]))
                                tar.makedir(member, dir_path)
                            else:
                                file_path=os.path.join(out_folder,os.path.sep.join(path[1:]))
                                tar.makefile(member,file_path)
                os.remove(model_file)
                logger.info("Model download done")
            return {f: os.path.sep.join([out_folder,f]) for f in os.listdir(out_folder)}
        except Exception as e:
            logger.error("Error downloading the model : {}".format(e))  # dispatch exception

    @classmethod
    def download_od_api_config(cls,pipeline_uri,out_file: str):
        try:
            assert validators.url(pipeline_uri),"invalid model uri param"
            assert FileUtil.exists_http_file(pipeline_uri),"The file doesn't exist"
            if not os.path.exists(out_file):
                logger.info("Downloading the pipeline file.....")
                FileUtil.download_file(pipeline_uri,out_file)
                logger.info("pipeline download done")
        except Exception as e:
            logger.error("Error downloading the model : {}".format(e))  # dispatch exception
