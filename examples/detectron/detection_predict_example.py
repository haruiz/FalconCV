from falconcv.cons import *
from falconcv.models import ModelBuilder
from falconcv.models.detectron import ModelZoo
from falconcv.util import VIUtil


def make_predictions(frozen_model, model_zoo_config, dataset_folder, labels_map, image):
    config = {
        "model": frozen_model,
        "model_zoo_config": model_zoo_config,
        "dataset_folder": dataset_folder,
        "labels_map": labels_map
    }
    # load freeze model
    with ModelBuilder.build(model=frozen_model, config=config, backend=DETECTRON) as model:
        img, predictions = model(image, threshold=0.5)
        import matplotlib
        matplotlib.use('TkAgg')
        VIUtil.imshow(img, predictions)


if __name__ == '__main__':
    dataset_folder = "<your dataset folder path>"
    frozen_model_file = "<your pth model folder path>"

    labels_map = {
        "Bird": 1,
        "Eagle": 2,
        "Falcon": 3
    }

    # pick model from zoo
    ModelZoo.print_available_models(task="detection")
    model_zoo_config = "R101"

    # make predictions
    image = "../images/birdtest.jpg"
    make_predictions(frozen_model_file, model_zoo_config, dataset_folder, labels_map, image)
