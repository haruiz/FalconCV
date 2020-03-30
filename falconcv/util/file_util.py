import configparser
import glob
import logging
import shutil
import zipfile
from urllib.parse import urlparse
import requests
import validators
from clint.textui import progress
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FileUtil:
    @classmethod
    def download_file(cls, file_uri, out_file) -> str:
        try:
            if not os.path.isfile(out_file):
                logger.debug("downloading file {}".format(file_uri))
                r = requests.get(file_uri, stream=True)
                with open(out_file, 'wb') as f:
                    total_length = int(r.headers.get('content-length'))
                    for chunk in progress.bar(
                            r.iter_content(chunk_size=1024),
                            expected_size=(total_length / 1024) + 1):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                logger.info("file downloaded successfully")
            return os.path.abspath(out_file)
        except Exception as e:
            logger.error("Error downloading the model : {}".format(e))

    @staticmethod
    def download_file_(uri: str, output_folder):
        try:
            if not validators.url(uri):
                raise Exception("Invalid URI {}".format(uri))
            output_file_name = os.path.basename(urlparse(uri).path)
            output_file_path = os.path.join(output_folder, output_file_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            if not os.path.exists(output_file_path):
                r = requests.get(uri, stream=True)
                with open(output_file_path, 'wb') as f:
                    size_file = int(r.headers.get('content-length'))
                    for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(size_file / 1024) + 1):
                        if chunk:
                            f.write(chunk)
                            f.flush()
            else:
                size_file = os.path.getsize(output_file_path)
            return output_file_path, size_file
        except Exception as e:
            logger.error("Error downloaing the file {}".format(e))

    @staticmethod
    def unzip_file(zip_file_path: str):
        zip_output_folder, _ = os.path.splitext(zip_file_path)
        if not os.path.exists(zip_output_folder):
            assert zipfile.is_zipfile(zip_file_path), "Invalid file format"
            with zipfile.ZipFile(zip_file_path, "r") as zf:
                zf.extractall(zip_output_folder)
        return zip_output_folder

    @classmethod
    def get_file(cls, file, folder):
        if validators.url(file):
            uri_path = urlparse(file).path
            file_name = os.path.basename(uri_path)
            files = cls.find_file(folder, file_name)
            if len(files) <= 0:
                files = [cls.download_file(file, folder)[0]]
            return files[0]
        else:
            files = cls.find_file(folder, os.path.basename(file))
            return files[0] if len(files) > 0 else None

    @staticmethod
    def find_file(folder: str, file_name: str):
        _, ext = os.path.splitext(file_name)
        return list(filter(lambda x: os.path.basename(x) == file_name,
                           glob.iglob("{}/**/*{}".format(folder, ext), recursive=True)))

    @staticmethod
    def exists_http_file(uri):
        assert validators.url(uri), "Invalid url format {}".format(uri)
        r = requests.get(uri)  # r=requests.head(uri)
        return r.status_code == requests.codes.ok  # check if the remote file exist

    @staticmethod
    def clear_folder(folder_path):
        for file_object in os.listdir(folder_path):
            file_object_path = os.path.join(folder_path, file_object)
            if os.path.isfile(file_object_path):
                os.unlink(file_object_path)
            else:
                shutil.rmtree(file_object_path)  # delete a folder tree

    @classmethod
    def delete_folder(cls, folder_path):
        if os.path.exists(folder_path):
            cls.clear_folder(folder_path)
            shutil.rmtree(folder_path)

    @classmethod
    def read_config(cls, config_path: str) -> dict:
        assert os.path.isfile(config_path), "File doesn't exist"
        config = configparser.ConfigParser()
        config.read(config_path)
        assert config.has_section("Model"), "Invalid config file"
        return dict(config["Model"])

    @staticmethod
    @contextmanager
    def workon(directory):
        owd = os.getcwd()
        try:
            os.chdir(directory)
            yield directory
        finally:
            os.chdir(owd)
