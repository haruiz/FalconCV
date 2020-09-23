from falconcv.util import VIUtil
from falconcv.cons import *
from falconcv.models import ModelBuilder
from falconcv.models.detectron import ModelZoo


def run_pretrained_model(pretrained_model, image):
    with ModelBuilder.build(model=pretrained_model, backend=DETECTRON) as model:
        img, predictions = model(image, threshold=0.7)
        import matplotlib
        matplotlib.use('TkAgg')
        VIUtil.imshow(img, predictions)


if __name__ == '__main__':
    # pick model from zoo
    ModelZoo.print_available_models(task="detection")

    # run pre-trained model
    pretrained_model = "R101"  # R101 / R50-FPN
    image = "../images/falcon.jpg"
    run_pretrained_model(pretrained_model, image)
    image = "../images/zebrahorse.png"
    run_pretrained_model(pretrained_model, image)
