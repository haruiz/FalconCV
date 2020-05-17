import sys

sys.path.append('.')

from falconcv.util import VIUtil
from falconcv.cons import *
from falconcv.models import ModelBuilder
from falconcv.models.detectron import DetectronModelZoo


def run_pretrained_model(pretrained_model, image):
    config = {
        "model": pretrained_model,
        "threshold": 0.6,
        "top_k": 10
    }

    with ModelBuilder.build(model=pretrained_model, config=config, backend=DETECTRON) as model:
        predictions = model.predict(image)
        VIUtil.img_show(image, predictions)


if __name__ == '__main__':
    # pick model from zoo
    DetectronModelZoo.print_available_models(task="detection")

    # run pre-trained model
    pretrained_model = "R101" # R101 / R50-FPN
    # image = "examples/images/falcon.jpg"
    image = "examples/images/zebrahorse.png"
    run_pretrained_model(pretrained_model, image)
