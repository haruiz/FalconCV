from falconcv.data.ds import OpenImages
from falconcv.models import ModelBuilder
from falconcv.models.tf import ModelZoo
from falconcv.util import FileUtil
from pathlib import Path
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
        for batch_images in dataset:
            for image in batch_images:
                image.export(data_folder)  # copy images to disk
    except Exception as ex:
        print(f"[ERROR] Error creating the dataset {ex}")


def train_model(model_name, images_folder, epochs=500):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder
        }
        with ModelBuilder.build(config=config) as model:
            model.train(epochs=epochs, val_split=0.3, clear_folder=True)
            model.freeze(epochs)
    except Exception as ex:
        raise Exception("Error training the model {} ".format(ex)) from ex


def make_predictions(model_folder, image):
    frozen_model = Path(model_folder).joinpath("export/frozen_inference_graph.pb")
    labels_map = Path(model_folder).joinpath("label_map.pbtxt")
    with ModelBuilder.build(str(frozen_model), str(labels_map)) as model:
        img, predictions = model(image, threshold=0.5)
        VIUtil.imshow(img, predictions)


if __name__ == '__main__':
    train_images_folder = "./data"
    model_name = "faster_rcnn_resnet50_coco"
    target_labels = ["bird", "eagle"]
    #create_detection_dataset(train_images_folder, target_labels, n_images=500, batch_size=50, split="train")
    #ModelZoo.print_available_models()
    make_predictions(f"./models/{model_name}", "https://nas-national-prod.s3.amazonaws.com/styles/hero_image/s3/web_groombaltimoreoriole-and-a-male-red-breasted-grosbeak.jpg")
    #train_model(model_name, train_images_folder)
