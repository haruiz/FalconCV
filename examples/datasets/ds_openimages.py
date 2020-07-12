import os

from falconcv.ds import OpenImages
from falconcv.util import FileUtil


def create_detection_dataset(images_folder, labels_map, n, batch_size, split):
    try:
        # creating dataset
        dataset = OpenImages(v=6)
        dataset.setup(split=split, task="detection")
        os.makedirs(images_folder, exist_ok=True)
        FileUtil.clear_folder(images_folder)
        for batch_images in dataset.fetch(
                n=n,
                labels=list(labels_map.keys()),
                batch_size=batch_size):
            for img in batch_images:
                img.export(images_folder, labels_map)
    except Exception as ex:
        print(f"[ERROR] Error creating the dataset {ex}")


if __name__ == '__main__':
    # images_folder = "<your images folder path>"
    train_images_folder = "/mnt/D/Dev/falconcv/datasets/openimages/animals/train"
    test_images_folder = "/mnt/D/Dev/falconcv/datasets/openimages/animals/test"
    labels_map = {
        "Bird": 1,
        "Eagle": 2,
        "Falcon": 3
    }
    # create detection the dataset for train
    create_detection_dataset(train_images_folder, labels_map, n=500, batch_size=250, split="train")

    # create detection the dataset for test
    create_detection_dataset(test_images_folder, labels_map, n=100, batch_size=100, split="test")
