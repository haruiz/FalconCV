import os

from falconcv.data.ds import Coco
from falconcv.models import ModelBuilder
from falconcv.models.tf import ModelZoo
from falconcv.util import FileUtil, ColorUtil
from pathlib import Path
from falconcv.util import VIUtil


def create_segmentation_dataset(images_folder, target_labels,color_palette, n_images, batch_size=10, split="train"):
    try:
        # creating dataset
        dataset = Coco(
            version=2017,  # versions 5 and 6 supported
            split=split,
            task="segmentation",
            labels=list(target_labels.keys()),  # target labels
            n_images=n_images,  # number of images by class
            batch_size=batch_size  # batch images size
        )
        print(len(dataset))  # size of dataset
        data_folder = Path(images_folder)
        data_folder.mkdir(exist_ok=True, parents=True)
        FileUtil.clear_folder(data_folder)
        # Download images
        for i, batch_images in enumerate(dataset):
            print(f"download done for batch {i+1} of {dataset.batches_count}")
            for image in batch_images:
                image.export(data_folder, labels_map, color_palette)  # copy images to disk
    except Exception as ex:
        print(f"[ERROR] Error creating the dataset {ex}")


def train_model(model_name, images_folder,model_folder, epochs=500):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder,
            "output_folder": model_folder,
        }
        with ModelBuilder.build(config=config) as model:
            #model.input_size = (300,300)
            model.batch_size = 1
            model.train(epochs=epochs, val_split=0.3, clear_folder=True)
            model.freeze(epochs)
    except Exception as ex:
        raise Exception("Error training the model {} ".format(ex)) from ex


def make_predictions(frozen_model,labels_map_file, image):
    # load freeze model
    with ModelBuilder.build(frozen_model,labels_map_file) as model:
        img,predictions=model(image, threshold=0.5)
        VIUtil.imshow(img,predictions)


if __name__ == '__main__':

    images_folder = "/home/haruiz/models/animals_seg/images"
    model_folder = "/home/haruiz/models/animals_seg/model"
    model_name = "mask_rcnn_resnet101_atrous_coco"
    labels_map = {
        "bird": 1,
        "cat": 2,
        "dog": 3
    }
    color_palette = ColorUtil.color_palette(n=len(labels_map))

    # create dataset
    #create_segmentation_dataset(images_folder,labels_map, color_palette,n_images=500, batch_size=100)

    # picking and training the model
    #print(ModelZoo.available_models(arch="mask"))  # check the models available
    train_model(model_name,images_folder, model_folder, epochs=1000)

    # doing inference
    #frozen_model_file = os.path.join(model_folder, "export/frozen_inference_graph.pb")
    #labels_map_file = os.path.join(model_folder, "label_map.pbtxt")
    #make_predictions(frozen_model_file, labels_map_file, "images/catsvrsbird2.jpeg")