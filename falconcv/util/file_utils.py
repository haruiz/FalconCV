import io
import math
import os
import shutil
import tarfile
import typing
import zipfile
from contextlib import contextmanager
from pathlib import Path
from stat import S_IRWXU

import logging
import requests
import validators
from urllib.parse import urlparse

from alive_progress import alive_bar


logger = logging.getLogger("rich")


class FileUtils:
    @staticmethod
    @contextmanager
    def workon(directory):
        directory = str(directory)
        owd = os.getcwd()
        try:
            os.chdir(directory)
            yield directory
        finally:
            os.chdir(owd)

    @staticmethod
    def get_file_size_from_uri(uri):
        return int(requests.head(uri).headers["Content-Length"])

    @staticmethod
    def get_file_size_from_path(path):
        # return os.path.getsize(path)
        return os.stat(path).st_size if path.exists() else 0

    @staticmethod
    def ls():
        return os.listdir(Path.cwd())

    @staticmethod
    def exists_http_file(uri):
        assert validators.url(uri), "Invalid url format {}".format(uri)
        r = requests.get(uri)
        return r.status_code == requests.codes.ok  # check if the remote file exist

    @classmethod
    def cp(cls, src: Path, dst: Path):
        shutil.copy(src, dst)

    @staticmethod
    def internet_on(url="http://www.google.com/", timeout=5):
        try:
            req = requests.get(url, timeout=timeout)
            req.raise_for_status()
            return True
        except requests.HTTPError as e:
            print(
                "Checking internet connection failed, status code {0}.".format(
                    e.response.status_code
                )
            )
        except requests.ConnectionError:
            print("No internet connection available.")
        return False

    @staticmethod
    def clear_folder(folder_path):
        for root, dirs, files in os.walk(folder_path):
            root = Path(root)
            for directory in dirs:
                os.chmod(root.joinpath(directory), S_IRWXU)
            for file in files:
                os.chmod(root.joinpath(file), S_IRWXU)
        shutil.rmtree(folder_path, ignore_errors=True)

    @staticmethod
    def prettysize(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    @classmethod
    def delete_folder(cls, folder_path):
        folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path
        if folder_path.exists():
            cls.clear_folder(folder_path)
            shutil.rmtree(folder_path)

    @classmethod
    def get_filename_from_uri_with_suffix(cls, file_uri):
        file_uri = str(file_uri)
        remote_file_path = urlparse(file_uri).path
        return Path(remote_file_path).name

    @classmethod
    def get_file_name_from_uri(cls, file_path):
        if validators.url(file_path):
            file_path = cls.get_filename_from_uri_with_suffix(file_path)
        file_path = Path(file_path)
        while file_path.suffix:
            file_path = file_path.with_suffix("")
        return file_path.name

    @classmethod
    def get_file_from_uri(
        cls,
        file_uri: str,
        out_folder: typing.Union[str, Path] = None,
        force: bool = False,
        unzip: bool = False,
        show_progress: bool = False,
    ):
        assert cls.internet_on(), "Not internet connection"
        assert validators.url(file_uri), "invalid file uri parameter"

        out_folder = Path(out_folder) if out_folder else Path(os.getcwd())
        out_folder.mkdir(exist_ok=True, parents=True)

        file_name_with_suffix = Path(cls.get_filename_from_uri_with_suffix(file_uri))
        file_name_with_non_suffix = Path(cls.get_file_name_from_uri(file_uri))
        download_file_path = out_folder.joinpath(file_name_with_suffix)
        download_unzip_path = out_folder.joinpath(file_name_with_non_suffix)

        is_compressed = file_name_with_suffix.suffix in [".zip", ".tar", ".gz"]
        out_path = (
            download_unzip_path if is_compressed and unzip else download_file_path
        )
        if out_path.exists() and not force:
            if is_compressed:
                return out_path
            remote_file_size = cls.get_file_size_from_uri(file_uri)
            locale_file_size = cls.get_file_size_from_path(out_path)
            if remote_file_size == locale_file_size:
                return out_path

        logger.info(f"downloading file : {file_uri} to {out_path}")
        if show_progress:
            r = requests.get(file_uri, stream=True)
            total_length = int(r.headers.get("content-length"))
            block_size = 1024  # chunk-size: 1 KBytes
            with open(download_file_path, "wb") as f, alive_bar(
                manual=True, title="Downloading File...."
            ) as bar:
                downloaded_bytes = 0
                for data in r.iter_content(block_size):
                    downloaded_bytes += len(data)
                    msg = f"{cls.prettysize(downloaded_bytes)} - {cls.prettysize(total_length)}"
                    bar(downloaded_bytes / total_length)
                    bar.text(msg)
                    f.write(data)
        else:
            r = requests.get(file_uri, stream=True)
            with open(download_file_path, "wb") as f:
                for data in r.iter_content():
                    f.write(data)
        if is_compressed and unzip:
            download_unzip_path.mkdir(exist_ok=True, parents=True)
            cls.unzip_file_to(download_file_path, download_unzip_path)
            download_file_path.unlink()
            return download_unzip_path
        return out_path

    @staticmethod
    def extract_files_from_zip(files_to_extract, zip_file_uri):
        file_name = FileUtils.get_file_name_from_uri(zip_file_uri)
        r = requests.get(zip_file_uri, stream=True)
        total_length = int(r.headers.get("content-length"))
        block_size = 1024  # chunk-size: 1 KBytes
        with io.BytesIO() as memory_file, alive_bar(
            manual=True, title=f"Loading file {file_name} in memory...."
        ) as bar:
            downloaded_bytes = 0
            for data in r.iter_content(block_size):
                downloaded_bytes += len(data)
                memory_file.write(data)
                msg = f"{FileUtils.prettysize(downloaded_bytes)} - {FileUtils.prettysize(total_length)}"
                bar(downloaded_bytes / total_length)
                bar.text(msg)

            extracted_files = {}
            with zipfile.ZipFile(memory_file) as zip_file:
                for file in files_to_extract:
                    with zip_file.open(file) as f:
                        extracted_files[file] = f.read()
            return extracted_files

    @staticmethod
    def read_csv(file_path, sep=","):
        with open(file_path, "r") as f:
            return [line.strip().split(sep) for line in f.readlines()]

    @staticmethod
    def unzip_file_to(file_path: typing.Union[str, Path], output_folder: Path):
        file_path = Path(file_path)
        assert file_path.exists(), f"file {file_path} not found"
        ext = file_path.suffix
        if ext == ".gz":
            with tarfile.open(file_path) as tar:
                dirs = [m for m in tar.getmembers() if m.isdir()]
                files = [m for m in tar.getmembers() if m.isfile()]
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
        elif ext == ".zip":
            assert zipfile.is_zipfile(file_path), "Invalid file format"
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(output_folder)
        else:
            raise IOError(f"Extension {ext} not supported yet")
