from falconcv.data.ds import Coco
from falconcv.util import FileUtil
from pathlib import Path


def create_detection_dataset(images_folder, target_labels, n_images, batch_size, split):
    try:
        # creating dataset
        dataset = Coco(
            version=2017,  # only 2017 version is supported
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


if __name__ == '__main__':
    # images_folder = "<your images folder path>"
    train_images_folder = "/mnt/D/Dev/falconcv/datasets/coco/animals/train"
    val_images_folder = "/mnt/D/Dev/falconcv/datasets/coco/animals/val"
    target_labels = ["bird", "eagle", "falcon"]
    # create detection the dataset for train
    create_detection_dataset(train_images_folder, target_labels, n_images=500, batch_size=250, split="train")

    # create detection the dataset for validation
    # create_detection_dataset(val_images_folder, target_labels, n_images=100, batch_size=100, split="validation")
