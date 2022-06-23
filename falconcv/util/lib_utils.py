from pathlib import Path
import falconcv
import importlib


class LibUtils:
    @classmethod
    def make_subfolder(cls, root_folder: Path, subfolder_path: str) -> Path:
        subfolder_dir = root_folder.joinpath(subfolder_path)
        subfolder_dir.mkdir(parents=True, exist_ok=True)
        return subfolder_dir

    @classmethod
    def datasets_home(cls, subfolder=None) -> Path:
        return cls.make_subfolder(falconcv.home(), f"datasets/{subfolder}")

    @classmethod
    def repos_home(cls) -> Path:
        return cls.make_subfolder(falconcv.home(), "repos")

    @classmethod
    def models_home(cls, subfolder) -> Path:
        return falconcv.home().joinpath(f"models/{subfolder}")

    @classmethod
    def pipelines_home(cls, subfolder) -> Path:
        return falconcv.home().joinpath(f"pipelines/{subfolder}")

    @staticmethod
    def try_import(package):
        try:
            return importlib.import_module(package)
        except ImportError:
            import pip

            pip.main(["install", package])
        finally:
            globals()[package] = importlib.import_module(package)
