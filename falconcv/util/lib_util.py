from pathlib import Path
import os
import importlib

class LibUtil:
    @staticmethod
    def home(root_folder=".falconcv"):
        usr_folder = str(Path.home())
        home_folder=os.path.sep.join([usr_folder,root_folder])
        os.makedirs(home_folder, exist_ok=True)
        return home_folder

    @classmethod
    def tf_models_home(cls):
        models_path =  os.path.sep.join([cls.home(), "models", "tf"])
        os.makedirs(models_path,exist_ok=True)
        return models_path

    @classmethod
    def tf_pipelines_home(cls):
        pipelines_folder =  os.path.sep.join([cls.home(), "pipelines", "tf"])
        os.makedirs(pipelines_folder,exist_ok=True)
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