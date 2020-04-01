import configparser
import glob
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import requests
import validators
from clint.textui import progress
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FileUtil:
    @staticmethod
    def download_file(file_uri: str, out_file: Path) -> Path:
        out_file = Path(out_file) \
            if isinstance(out_file, str) else out_file
        if not out_file.exists():
            logger.debug("downloading file {}".format(file_uri))
            r = requests.get(file_uri, stream=True)
            with open(str(out_file), 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(
                        r.iter_content(chunk_size=1024),
                        expected_size=(total_length / 1024) + 1):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            logger.info("file downloaded successfully")
        return  out_file

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

    @classmethod
    def download_and_unzip_file(cls, file_uri, output_folder: Path):
        assert validators.url(file_uri), "invalid file uri parameter"
        output_folder = Path(output_folder) \
            if isinstance(output_folder, str) else output_folder
        output_folder.mkdir(exist_ok=True)
        filename = Path(urlparse(file_uri).path).name
        downloaded_file = cls.download_file(file_uri, output_folder.joinpath(filename))
        ext = downloaded_file.suffix
        if ext == ".gz":
            with tarfile.open(str(downloaded_file)) as tar:
                dirs: [tarfile.TarInfo] = [m for m in tar.getmembers() if m.isdir()]
                files: [tarfile.TarInfo] = [m for m in tar.getmembers() if m.isfile()]
                for member in dirs:
                    path_parts = Path(member.name).parts
                    if len(path_parts) > 1:
                        folder_path = os.sep.join(Path(member.name).parts[1:])
                        folder_path = output_folder.joinpath(folder_path)
                        tar.makedir(member, folder_path)

                for member in files:
                    file_path = os.sep.join(Path(member.name).parts[1:])
                    file_path = output_folder.joinpath(file_path)
                    tar.makefile(member, file_path)
            downloaded_file.unlink()
        else:
            raise IOError("Extension not supported yet") # for other kind of file extensions like .zip

