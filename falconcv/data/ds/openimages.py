import logging

import boto3
import dask
import dask.dataframe as dd
import more_itertools
import numpy as np
from dask.diagnostics import ProgressBar

from falconcv.util import S3Util
from .image import TaggedImage, BoxRegion
from .remote_dataset import RemoteDataset
logger = logging.getLogger(__name__)


class OpenImages(RemoteDataset):
    def __init__(self, version: int = 2017, task="detection", split: str = "train", labels: [str] = None,
                 n_images: int = 0,
                 batch_size: int = 12, is_truncated=False, is_depiction=False, is_occluded=False):
        """
        Create an instance of the openImages dataset
        :param v: version of the dataset
        :param labels: labels to download
        :param count: number of images by label
        :param batch_size: number of images by batch        """
        assert version == 5 or version == 6, "version not supported"
        assert task == "detection", "Invalid task parameter"
        assert split in ["train", "test", "validation"], "invalid split parameter"
        super(OpenImages, self).__init__(version, split, labels, n_images, batch_size)
        self.seg_available_labels = {}

        self._is_truncated = is_truncated
        self._is_depiction = is_depiction
        self._is_occluded = is_occluded
        self._task = task
        self._s3 = boto3.resource('s3')
        if version == 5:
            self._files = {
                "class_names_object_detection": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
                "class_names_segmentation": "https://storage.googleapis.com/openimages/v5/classes-segmentation.txt",

                "train_images": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
                "test_images": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
                "validation_images": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",

                "train_annotations_bbox": "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv",
                "test_annotations_bbox": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
                "validation_annotations_bbox": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
                # "train_annotations_masks": "https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv",
                # "test_annotations_masks": "https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv",
                # "validation_annotations_mask": "https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv"
            }
        else:
            self._files = {
                "class_names_object_detection": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
                "class_names_segmentation": "https://storage.googleapis.com/openimages/v5/classes-segmentation.txt",

                "train_images": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
                "test_images": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
                "validation_images": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",

                "train_annotations_bbox": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
                "test_annotations_bbox": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
                "validation_annotations_bbox": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
                # "train_annotations_masks": "https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv",
                # "test_annotations_masks": "https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv",
                # "validation_annotations_mask": "https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv"
            }
        self.__load__()

    def _get_annotation_file(self):
        dep_map = {
            "detection": {
                "train": self._get_dependency("train_annotations_bbox"),
                "validation": self._get_dependency("validation_annotations_bbox"),
                "test": self._get_dependency("test_annotations_bbox")
            },
            "segmentation": {
                "train": self._get_dependency("train_annotations_masks"),
                "validation": self._get_dependency("validation_annotations_mask"),
                "test": self._get_dependency("test_annotations_masks")
            }
        }
        return dep_map[self._task][self._split]

    def __getbatch__(self, batch):
        delayed_tasks = []
        for image_id, image_rois in batch:
            download_task = dask.delayed(self._fetch_single_image(
                image_id,
                image_rois
            ))
            delayed_tasks.append(download_task)
        with ProgressBar():
            results = dask.compute(*delayed_tasks)
        results = [img for img in results if img]
        return results

    def _fetch_single_image(self, image_id, parts):
        try:

            bucket = "open-images-dataset"
            key = "{}/{}.jpg".format(self._split, image_id)
            arr = S3Util.fetch_image_unsigned(bucket, key)
            if isinstance(arr, np.ndarray):
                tagged_image = TaggedImage(arr)
                tagged_image.id = image_id
                if parts.empty:
                    return None
                self._annotate_image(tagged_image, parts, self._task)
                return tagged_image
        except Exception as ex:
            logger.error("error downloading the image with id {} : {}".format(image_id, ex))
        return None

    def _annotate_image(self, tagged_image: TaggedImage, image_parts: dd.DataFrame, task: str):
        try:
            image: np.ndarray = tagged_image.img
            h = np.size(image, 0)
            w = np.size(image, 1)
            if task == "detection":
                for _, row in image_parts.iterrows():
                    box = BoxRegion()
                    box.shape_attributes["x"] = int(row["XMin"] * w)
                    box.shape_attributes["y"] = int(row["YMin"] * h)
                    box.shape_attributes["width"] = int(row["XMax"] * w) - box.shape_attributes["x"]
                    box.shape_attributes["height"] = int(row["YMax"] * h) - box.shape_attributes["y"]
                    box.region_attributes["name"] = self._available_labels[row["LabelName"]]
                    box.region_attributes["is_occluded"] = row["IsOccluded"]
                    box.region_attributes["is_truncated"] = row["IsTruncated"]
                    box.region_attributes["is_group_of"] = row["IsDepiction"]
                    box.region_attributes["is_depiction"] = row["IsInside"]
                    tagged_image.regions.append(box)
        except Exception as ex:
            logger.error("error annotating the image with id {} : {}".format(tagged_image.id, ex))

    def __load__(self):
        try:
            self._download_dependencies()
            class_descriptions_csv = self._get_dependency("class_names_object_detection")
            classes_segmentation_txt = self._get_dependency("class_names_segmentation")
            if class_descriptions_csv and classes_segmentation_txt:
                with ProgressBar():
                    dds = dd.read_csv(class_descriptions_csv).compute()
                    for index, row in dds.iterrows():
                        label_id = row[0]
                        label_name = row[1]
                        self._available_labels[label_id] = label_name
                    dds = dd.read_csv(classes_segmentation_txt).compute()
                    self.seg_available_labels = {row[0]: self.available_labels[row[0]] for _, row in dds.iterrows() if
                                                 row[0] in self._available_labels}

                valid_labels = self._valid_labels(self._labels)  # obtain valid labels
                if valid_labels:
                    annotations_file = self._get_annotation_file()  # get annotations file path
                    ann_df = dd.read_csv(annotations_file, assume_missing=True)  # read annotations files
                    for class_id, class_name in valid_labels.items():
                        # obtain annotations for each file
                        ddc = ann_df[
                            (ann_df["LabelName"] == class_id) &
                            (ann_df["IsTruncated"] == self._is_truncated) &
                            (ann_df['IsDepiction'] == self._is_depiction) &
                            (ann_df['IsOccluded'] == self._is_occluded)]
                        images = ddc.compute().groupby("ImageID")
                        if self._n_images: self._images += more_itertools.take(self._n_images, images)

        except Exception as ex:
            logger.error("Error loading the dataset : {} ".format(ex))
            raise ex

    def _valid_labels(self, labels: []):
        """
        @:return Check  if a set of labels is valid and return them
        """
        labels_map = self._available_labels if self._task == "detection" or "classification" else self.seg_available_labels
        if labels is None:
            return labels_map
        valid_labels = list(map(lambda l: l.capitalize(), labels))
        valid_labels = {k: v for k, v in labels_map.items() if v in valid_labels}
        return valid_labels



