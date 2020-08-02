import os
from falconcv.data import Coco
from falconcv.models import ModelBuilder
from falconcv.util import FileUtil, ImageUtil, VIUtil, ColorUtil
from falconcv.models.tf import ModelZoo

def download_data(labels_map, color_palette, n_images, batch_size, split, task, output_folder):
    try:
        # creating dataset
        dataset = Coco(v=2017)
        dataset.setup(split=split, task=task)
        os.makedirs(output_folder, exist_ok=True)
        FileUtil.clear_folder(output_folder)
        for batch_images in dataset.fetch(
                n=n_images,
                labels=list(labels_map.keys()),
                batch_size=batch_size):
            for img in batch_images:
                img.export(output_folder, labels_map, color_palette)
    except Exception as ex:
        print(f"Error descargando dataset: {ex}")


def train_and_freeze_model(model_name, images_folder, out_folder, labels_map, epochs=5000):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder,
            "output_folder": out_folder,
            "labels_map": labels_map,
        }
        with ModelBuilder.build(config=config) as model:
            model.train(epochs=epochs, val_split=0.3, clear_folder=True)
            #model.freeze(epochs)

    except Exception as ex:
        raise Exception(f"Error training the model {ex}") from ex


def create_dataset():

    color_palette = ColorUtil.color_palette(n=len(labels_map))
    # download train images
    download_data(labels_map=labels_map, color_palette=color_palette, n_images=500,
                  batch_size=250, split="train", task="segmentation", output_folder="./data/train")
    # download validation images
    download_data(labels_map=labels_map, color_palette=color_palette, n_images=200,
                  batch_size=100, split="validation", task="segmentation", output_folder="./data/val")



if __name__ == '__main__':
    #ModelZoo.print_available_models(arch="mask")
    labels_map = {
        "airplane": 1,
        "train": 2
    }
    images_folder = "./data/train/"
    model_folder = "./models/tf"
    train_and_freeze_model("mask_rcnn_inception_v2_coco", images_folder, model_folder, labels_map)