from pathlib import Path
import importlib


class LibUtil:
    @staticmethod
    def home(root_folder=".falconcv") -> Path:
        usr_folder = Path.home()
        home_folder=usr_folder.joinpath(root_folder)
        home_folder.mkdir(exist_ok=True)
        return home_folder

    @classmethod
    def models_home(cls, subfolder)-> Path:
        models_path = cls.home().joinpath("models/{}".format(subfolder))
        models_path.mkdir(exist_ok=True)
        return models_path

    @classmethod
    def datasets_home(cls) -> Path:
        datasets_path = cls.home().joinpath("datasets")
        datasets_path.mkdir(exist_ok=True)
        return datasets_path

    @classmethod
    def repos_home(cls) -> Path:
        repos_path = cls.home().joinpath("repos")
        repos_path.mkdir(exist_ok=True)
        return repos_path

    @classmethod
    def pipelines_home(cls, subfolder) -> Path:
        pipelines_folder = cls.home().joinpath("pipelines/{}".format(subfolder))
        pipelines_folder.mkdir(exist_ok=True)
        return pipelines_folder

    @staticmethod
    def try_import(package):
        try:
            return importlib.import_module(package)
        except ImportError:
            import pip
            pip.main(['install',package])
        finally:
            globals()[package]=importlib.import_module(package)