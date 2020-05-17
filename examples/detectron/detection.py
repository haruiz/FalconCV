import sys
sys.path.append('.')

from falconcv.util import VIUtil
from falconcv.cons import *
from falconcv.models import ModelBuilder
from falconcv.models.detectron import DetectronModelZoo


def run_pretrained_model(pretrained_model, image, task="detection"):
    with ModelBuilder.build(pretrained_model, backend=DETECTRON) as model:
        output = model.predict(image, threshold=0.5)
        VIUtil.img_show(image, output)


if __name__ == '__main__':
    # pick model from zoo
    DetectronModelZoo.print_available_models(task="detection")

    # run pre-trained model
    pretrained_model = "R50-FPN"
    image = "examples/images/falcon.jpg"
    run_pretrained_model(pretrained_model, image, task="detection")
