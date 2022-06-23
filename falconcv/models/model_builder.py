from falconcv.cons import *


class ModelBuilder:
    @classmethod
    def build(
        cls, save_model_path=None, label_map_path=None, config=None, backend=TODA
    ):
        if backend == TODA:
            from falconcv.models.toda.toda_installer import TODAInstaller
            TODAInstaller().install()

            from falconcv.models.toda.toda_trained_model import TODATrainedModel
            from falconcv.models.toda.toda_trainable_model import TODATrainableModel

            if config:
                return TODATrainableModel.from_config(config)
            else:
                return TODATrainedModel(save_model_path, label_map_path)

        elif backend == DETECTRON:
            raise NotImplementedError("Not implemented yet")
        else:
            raise NotImplementedError("Invalid backend parameter")
