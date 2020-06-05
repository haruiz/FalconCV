import logging
import math
import os

import boto3
import dask
import dask.dataframe as dd
import more_itertools
import numpy as np
import cv2
from dask.diagnostics import ProgressBar

from falconcv.ds.dataset import DatasetDownloader
from falconcv.ds.image import TaggedImage,BoxRegion,PolygonRegion
from falconcv.util import S3Util

logger=logging.getLogger(__name__)


class OpenImages(DatasetDownloader):

    def __init__(self, v = 6):
        super(OpenImages,self).__init__()
        assert v == 5 or v == 6, "version not supported"
        self._s3=boto3.resource('s3')
        if v== 5 :
            self._remote_dep = {
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
            self._remote_dep = {
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

    def setup(self, split="train", task="detection"):
        try:
            assert task == "detection","Invalid task parameter"
            assert split in ["train","test","validation"],"invalid split parameter"
            super(OpenImages,self).setup(split, task)
            class_descriptions_csv=self._get_dependency("class_names_object_detection")
            classes_segmentation_txt=self._get_dependency("class_names_segmentation")
            if class_descriptions_csv and classes_segmentation_txt:
                with ProgressBar():
                    dds=dd.read_csv(class_descriptions_csv).compute()
                    for index,row in dds.iterrows():
                        label_id=row[0]
                        label_name=row[1]
                        self.labels_map[label_id]=label_name
                    dds=dd.read_csv(classes_segmentation_txt).compute()
                    self.slabels_map ={row[0]: self.labels_map[row[0]] for _,row in dds.iterrows() if row[0] in self.labels_map}
            return self
        except Exception as ex:
            logger.error("Error preparing the dataset : {} ".format(ex))
            raise ex

    def _get_annotation_file(self):
        dep_map = {
            "detection":{
                "train": self._get_dependency("train_annotations_bbox"),
                "validation": self._get_dependency("validation_annotations_bbox"),
                "test": self._get_dependency("test_annotations_bbox")
            },
            "segmentation":{
                "train": self._get_dependency("train_annotations_masks"),
                "validation": self._get_dependency("validation_annotations_mask"),
                "test": self._get_dependency("test_annotations_masks")
            }
        }
        return dep_map[self.task][self.split]

    def _valid_labels(self,labels: []):
        """@:return Check  if a set of labels is valid and return them"""
        labels_map=self.labels_map if self.task == "detection" or "classification" else self.slabels_map
        if labels is None:
            return labels_map
        valid_labels=list(map(lambda l: l.capitalize(),labels))
        valid_labels={k: v for k,v in labels_map.items() if v in valid_labels}
        return valid_labels

    @dask.delayed
    def _fetch_single_image(self,image_id,parts):
        try:
            bucket="open-images-dataset"
            key = "{}/{}.jpg".format(self.split,image_id)
            arr = S3Util.fetch_image_unsigned(bucket,key)
            if isinstance(arr, np.ndarray):
                tagged_image=TaggedImage(arr)
                tagged_image.id=image_id
                if parts.empty:
                    return None
                self._annotate_image(tagged_image,parts,self.task)
                return tagged_image
        except Exception as ex:
            logger.error("error downloading the image with id {} : {}".format(image_id,ex))
        return None

    def _annotate_image(self, tagged_image: TaggedImage, image_parts: dd.DataFrame, task: str):
        try:
            image: np.ndarray=tagged_image.img
            h=np.size(image,0)
            w=np.size(image,1)
            if task == "detection":
                for _, row in image_parts.iterrows():
                    box = BoxRegion()
                    box.shape_attributes["x"] = int(row["XMin"]*w)
                    box.shape_attributes["y"] = int(row["YMin"]*h)
                    box.shape_attributes["width"] = int(row["XMax"]*w)-box.shape_attributes["x"]
                    box.shape_attributes["height"] = int(row["YMax"]*h)-box.shape_attributes["y"]
                    box.region_attributes["name"] = self._labels_map[row["LabelName"]]
                    box.region_attributes["is_occluded"] = row["IsOccluded"]
                    box.region_attributes["is_truncated"]= row["IsTruncated"]
                    box.region_attributes["is_group_of"]=  row["IsDepiction"]
                    box.region_attributes["is_depiction"]= row["IsInside"]
                    tagged_image.regions.append(box)
        except Exception as ex:
            logger.error("error annotating the image with id {} : {}".format(tagged_image.id,ex))

    def fetch(self,
              n = None,
              labels=None,
              batch_size: int = 200):
        try:
            valid_labels=self._valid_labels(labels)
            annotations_file=self._get_annotation_file()
            ann_df=dd.read_csv(annotations_file,assume_missing=True)
            for class_id,class_name in valid_labels.items():
                logger.info("downloading images for : {}".format(class_name))
                ddc = ann_df[(ann_df["LabelName"] == class_id)]
                images = ddc.compute().groupby("ImageID")
                if n: images = more_itertools.take(n, images)
                number_of_batches = math.ceil(len(images) / batch_size)
                for i, batch in  enumerate(more_itertools.chunked(images, batch_size)):
                    delayed_tasks=[]
                    for image_id,image_rois in batch:
                        download_task=self._fetch_single_image(
                            image_id,
                            image_rois
                        )
                        delayed_tasks.append(download_task)
                    logger.info("downloading batch {}/{}".format(i+1, number_of_batches))
                    if len(delayed_tasks) > 0:
                        results=dask.compute(*delayed_tasks)
                        results = [img for img in results if img]
                        yield results
                        del results
        except Exception as ex:
            logger.exception("Error fetching the images : {} ".format(ex)) # in case something wrong happens
            raise  ex

