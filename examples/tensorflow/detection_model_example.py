import os

from pathlib import Path

from falconcv.data.ds import OpenImages
from falconcv.models import ModelBuilder
from falconcv.models.tf import ModelZoo
from falconcv.util import FileUtil
from falconcv.util import VIUtil


def create_detection_dataset(images_folder, target_labels, n_images, batch_size, split):
    try:
        # creating dataset
        dataset = OpenImages(
            version=6,  # versions 5 and 6 supported
            split=split,
            task="detection",
            labels=target_labels,  # target labels
            n_images=n_images,  # number of images by class
            batch_size=batch_size  # batch images size
        )
        print(len(dataset))  # size of dataset
        data_folder = Path(images_folder)
        data_folder.mkdir(exist_ok=True)
        FileUtil.clear_folder(data_folder)
        # Download images
        for i, batch_images in enumerate(dataset):
            print(f"[INFO] Download done for batch {i + 1} of {dataset.batches_count}")
            for image in batch_images:
                image.export(data_folder)  # copy images to disk
    except Exception as ex:
        print(f"[ERROR] Error creating the dataset {ex}")


def train_model(model_name, images_folder, model_folder, epochs=500):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder,
            "output_folder": model_folder,
        }
        with ModelBuilder.build(config=config) as model:
            model.batch_size = 1
            model.train(epochs=epochs, val_split=0.3, clear_folder=True)
            model.freeze(epochs)
    except Exception as ex:
        raise Exception("[ERROR] Error training the model {} ".format(ex)) from ex


def make_predictions(frozen_model, labels_map_file, image):
    # load freeze model
    with ModelBuilder.build(frozen_model, labels_map_file) as model:
        img, predictions = model(image, threshold=0.5)
        VIUtil.imshow(img, predictions)


if __name__ == '__main__':
    images_folder = "./data"
    model_folder = "./mymodel"

    target_labels = ["bird", "eagle", "falcon"]

    # create dataset
    create_detection_dataset(images_folder, target_labels, n_images=500, batch_size=100)

    # picking and training the model
    ModelZoo.print_available_models(arch="faster")  # check the models available
    model_name = "faster_rcnn_resnet50_coco"
    train_model(model_name, images_folder, model_folder, epochs=5000)

    # doing inference
    frozen_model_file = os.path.join(model_folder, model_name, "export/frozen_inference_graph.pb")
    labels_map_file = os.path.join(model_folder, model_name, "label_map.pbtxt")
    make_predictions(frozen_model_file, labels_map_file, "../images/catsvrsbird2.jpeg")
