from pathlib import Path

from falconcv.data.ds import OpenImages
from falconcv.util import FileUtil


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


if __name__ == '__main__':
    images_folder = "<your images folder path>"
    target_labels = ["bird", "eagle", "falcon"]

    # create detection the dataset for train split
    create_detection_dataset(images_folder, target_labels, n_images=500, batch_size=50, split="train")
