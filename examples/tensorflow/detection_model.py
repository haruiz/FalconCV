import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from falconcv.models.tf import ModelZoo
from falconcv.ds import OpenImages
from falconcv.util import FileUtil, VIUtil
from falconcv.models import ModelBuilder


def create_dataset(images_folder, labels_map, n, batch_size):
    try:
        # creating dataset
        dataset = OpenImages(v=6)
        dataset.setup(split="train", task="detection")
        # labels = dataset.labels_map.values() # get valid labels
        os.makedirs(images_folder, exist_ok=True)
        FileUtil.clear_folder(images_folder)
        for batch_images in dataset.fetch(
                n=n,
                labels=list(labels_map.keys()),
                batch_size=batch_size):
            for img in batch_images:
                img.export(images_folder, labels_map)
                for region in img.regions:
                    pass
                    # print(region.shape_attributes["x"],
                    #       region.shape_attributes["y"])
    except Exception as ex:
        print("error creating the dataset {} ".format(ex))


def train_model(model_name, images_folder, out_folder, labels_map, epochs=2000):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder,
            "output_folder": out_folder,
            "labels_map": labels_map,
        }
        with ModelBuilder.build(config=config) as model:
            model.train(epochs=epochs, val_split=0.3, clear_folder=True)
            model.freeze(epochs)
    except Exception as ex:
        raise Exception("Error training the model {} ".format(ex)) from ex


def make_predictions(frozen_model, labels_map_file, image):
    # load freeze model
    with ModelBuilder.build(frozen_model, labels_map_file) as model:
        img, predictions = model.predict(image, threshold=0.5)
        #import matplotlib
        #matplotlib.use('WXAgg')
        VIUtil.imshow(img, predictions)


if __name__ == '__main__':
    images_folder = "<your images folder path>"
    model_folder = "<your model folder path>"
    labels_map = {
        "Bird": 1,
        "Eagle": 2,
        "Falcon": 3
    }
    # create the dataset
    create_dataset(images_folder, labels_map, n=500, batch_size=250)

    # pick model from zoo
    ModelZoo.print_available_models(arch="faster")

    # train model
    model = "faster_rcnn_inception_v2_coco"
    train_model(model, images_folder, model_folder, labels_map)

    # inference
    frozen_model_file = os.path.join(model_folder, model, "export/frozen_inference_graph.pb")
    labels_map_file = os.path.join(model_folder, model, "label_map.pbtxt")
    from glob import glob
    for image in glob("../../examples/images/*"):
        make_predictions(frozen_model_file, labels_map_file, image)
