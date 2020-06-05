import configparser
import glob
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import requests
import typing
import validators
from clint.textui import progress
import os
from contextlib import contextmanager
from tqdm import tqdm
from falconcv.decor import pathassert

logger = logging.getLogger(__name__)


class FileUtil:

    @staticmethod
    def internet_on(url='http://www.google.com/', timeout=5):
        try:
            req = requests.get(url, timeout=timeout)
            req.raise_for_status()
            return True
        except requests.HTTPError as e:
            print("Checking internet connection failed, status code {0}.".format(
                e.response.status_code))
        except requests.ConnectionError:
            print("No internet connection available.")
        return False

    @classmethod
    def download_file(cls, file_uri: str, out_folder: typing.Union[str, Path] = None, force=False, unzip=True, show_progress=False):
        try:
            assert cls.internet_on(), "Not internet connection"
            assert validators.url(file_uri), "invalid file uri parameter"
            out_folder = Path(os.getcwd()) if out_folder is None else out_folder
            out_folder = out_folder if isinstance(out_folder, Path) else Path(out_folder)
            out_folder.mkdir(exist_ok=True)
            # get remote file name
            remote_file_path = urlparse(file_uri).path
            remote_file_name = Path(remote_file_path).name
            out_file = out_folder.joinpath(remote_file_name)
            # dont download the file if it already exists
            if not out_file.exists() or force:
                logger.debug("[INFO]: downloading file : {}".format(file_uri))
                if show_progress:
                    r = requests.get(file_uri, stream=True)
                    total_length = int(r.headers.get('content-length'))
                    block_size = 1024  # 1 Kibibyte
                    # disable=True for unit tests
                    with tqdm(total=total_length,
                              unit='iB',
                              unit_scale=True,
                              desc="downloading file {}".format(remote_file_name)) as t:
                        with open(str(out_file), 'wb') as f:
                            for data in r.iter_content(block_size):
                                t.update(len(data))
                                f.write(data)
                else:
                    r = requests.get(file_uri, stream=True)
                    with open(str(out_file), 'wb') as f:
                        for data in r.iter_content():
                            f.write(data)

                logger.debug("[INFO]: File ({}) download done".format(file_uri))
            if unzip and out_file.suffix in [".gz", ".zip"]:
                #unzip_out_folder_name = remote_file_name[:remote_file_name.find('.')]
                #unzip_out_folder = out_folder.joinpath(unzip_out_folder_name)
                cls.unzip_file(out_file, out_folder)
            return out_file
        except Exception as ex:
            raise ex

    @staticmethod
    @contextmanager
    def workon(directory):
        owd = os.getcwd()
        try:
            os.chdir(directory)
            yield directory
        finally:
            os.chdir(owd)

    @staticmethod
    def exists_http_file(uri):
        assert validators.url(uri), "Invalid url format {}".format(uri)
        r = requests.get(uri)  # r=requests.head(uri)
        return r.status_code == requests.codes.ok  # check if the remote file exist

    @staticmethod
    def clear_folder(folder_path):
        folder_path = Path(folder_path) \
            if isinstance(folder_path, str) else folder_path
        for path in folder_path.iterdir():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)

    @classmethod
    def delete_folder(cls, folder_path):
        folder_path = Path(folder_path) \
            if isinstance(folder_path, str) else folder_path
        if folder_path.exists():
            cls.clear_folder(folder_path)
            shutil.rmtree(folder_path)

    @staticmethod
    @pathassert
    def unzip_file(file_path: typing.Union[str, Path], output_folder: Path):
        ext = file_path.suffix
        if ext == ".gz":
            with tarfile.open(str(file_path)) as tar:
                dirs: [tarfile.TarInfo] = [m for m in tar.getmembers() if m.isdir()]
                files: [tarfile.TarInfo] = [m for m in tar.getmembers() if m.isfile()]
                # unzip the dirs
                for member in dirs:
                    path_parts = Path(member.name).parts
                    # ignore root folder
                    if len(path_parts) > 1:
                        tar_folder_path = os.sep.join(Path(member.name).parts[1:])
                        tar_folder_path = output_folder.joinpath(tar_folder_path)
                        tar.makedir(member, tar_folder_path)
                # unzip the files
                for member in files:
                    tar_file_path = os.sep.join(Path(member.name).parts[1:])
                    tar_file_path = output_folder.joinpath(tar_file_path)
                    tar.makefile(member, tar_file_path)
            file_path.unlink()
        elif ext == ".zip":
            assert zipfile.is_zipfile(file_path), "Invalid file format"
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(output_folder)
        else:
            raise IOError("Extension {} not supported yet".format(ext))


