import io
import itertools
import typing
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageDraw
from alive_progress import alive_bar
from dask import dataframe as dd

from falconcv.data import PascalVOCImage
from falconcv.data.remote_dataset import RemoteDataset
from falconcv.decor import timeit
from falconcv.util import FileUtils, DatasetUtils, ColorUtils, S3Utils
import logging

logger = logging.getLogger("rich")


class OpenImages(RemoteDataset):
    def __init__(
        self,
        version: int = 6,
        split: str = "train",
        task: str = "detection",
    ):
        super().__init__()
        self._version = version
        self._split = split
        self._task = task
        self._s3 = boto3.resource("s3")
        self.__load__()

    @property
    def available_labels(self):
        classes_desc_file = self._get_dependency("class_names")
        class_names_file_rows = FileUtils.read_csv(classes_desc_file)
        labels_map = {row[0]: row[1].lower() for row in class_names_file_rows}
        if self._task == "segmentation":
            classes_desc_file = self._get_dependency("segmentation_classes")
            seg_class_names_file_rows = FileUtils.read_csv(classes_desc_file)
            labels_map = {
                row[0]: labels_map[row[0]] for row in seg_class_names_file_rows
            }
        return labels_map

    def __load__(self):
        self._deps_map = DatasetUtils.get_openimages_deps(self._version, self._split)
        self._download_dependencies()

    def fetch(
        self,
        labels: typing.List[str] = None,
        n_images: int = None,
        batch_size: int = None,
    ):
        labels_map = self._get_valid_labels(labels)
        assert labels_map, "Valid labels not found"
        labels_map_color_palette = ColorUtils.color_palette(n=len(labels_map))
        labels_id = {k: i + 1 for i, k in enumerate(labels_map.keys())}

        images = self._search_images(labels_map, n_images)
        batch_size = batch_size or len(images)
        batch_num = int(np.ceil(len(images) / batch_size))
        current_batch = 1
        for i in range(0, len(images), batch_size):
            logger.info(f"Processing batch {current_batch} / {batch_num}")
            yield self._get_images_batch(
                images[i : i + batch_size],
                labels_map,
                labels_id,
                labels_map_color_palette,
            )
            current_batch += 1

    def _get_valid_labels(self, labels):
        labels = list(map(str.lower, labels))
        available_labels = {v: k for k, v in self.available_labels.items()}
        valid_labels = {
            available_labels[label]: label
            for label in labels
            if label in available_labels
        }
        return valid_labels

    # @timeit
    def _search_images(self, valid_labels, n_images, **kwargs):
        images = []
        for label, label_id in valid_labels.items():
            if self._task == "segmentation":
                images += self._search_image_for_segmentation(label, n_images, **kwargs)
            else:
                images += self._search_image_for_detection(label, n_images, **kwargs)
        return images

    def _search_image_for_detection(self, label, n_images, **kwargs):
        box_annotations_file = self._get_dependency("boxes")
        ann_df = dd.read_csv(box_annotations_file, assume_missing=True)
        ann_df = ann_df[
            (ann_df["LabelName"] == label)
            & (ann_df["IsTruncated"] == kwargs.get("IsTruncated", False))
            & (ann_df["IsDepiction"] == kwargs.get("IsDepiction", False))
            & (ann_df["IsOccluded"] == kwargs.get("IsOccluded", False))
        ].compute()
        ann_df = list(ann_df.groupby("ImageID"))
        ann_df = ann_df[:n_images]
        return ann_df

    def _search_image_for_segmentation(self, label, n_images, **kwargs):
        masks_annotations_file = self._get_dependency("segmentation")
        ann_df = dd.read_csv(masks_annotations_file, assume_missing=True)
        columns = {col: col[3:] for col in ann_df.columns if col.startswith("Box")}
        ann_df = ann_df.rename(columns=columns)
        ann_df = ann_df[
            (ann_df["LabelName"] == label)
            & (ann_df["PredictedIoU"] >= kwargs.get("PredictedIoU", 0.5))
        ].compute()
        ann_df = list(ann_df.groupby("ImageID"))
        ann_df = ann_df[:n_images]
        return ann_df

    def _get_images_batch(
        self, images_batch, labels_map, labels_id, labels_color_palette
    ):
        images_batch_annotations = dict(images_batch)
        images_batch_images = self._get_images_batch_data(
            self._split, images_batch_annotations
        )
        images_batch_masks = None
        if self._task == "segmentation":
            images_batch_masks = self._get_masks_batch_data(images_batch_annotations)
        images = []
        for image_id, img_arr in images_batch_images.items():
            img = PascalVOCImage()
            img.image_arr = img_arr
            w, h = img_arr.size
            ann_df = images_batch_annotations[image_id]
            labels_mask = None
            for _, row in ann_df.iterrows():
                img.add_box(
                    labels_map[row["LabelName"]],
                    int(row["XMin"] * w),
                    int(row["YMin"] * h),
                    int(row["XMax"] * w),
                    int(row["YMax"] * h),
                )
                if images_batch_masks is not None:
                    if labels_mask is None:
                        labels_mask = PILImage.new("P", (w, h), 0)
                        labels_mask.putpalette(labels_color_palette)
                    object_mask = images_batch_masks[row["MaskPath"]]
                    object_mask = object_mask.resize((w, h))
                    label_id = labels_id[row["LabelName"]]
                    drawable_image = ImageDraw.Draw(labels_mask)
                    drawable_image.bitmap((0, 0), object_mask, fill=label_id)
                    del drawable_image
            if labels_mask is not None:
                img.mask_arr = labels_mask
            images.append(img)
        return images

    @classmethod
    def _get_images_batch_data(cls, split, images_batch) -> dict:
        with ThreadPoolExecutor(max_workers=len(images_batch)) as executor:
            futures = {
                executor.submit(cls._fetch_image, split, image_id): image_id
                for image_id, _ in images_batch.items()
            }
            images_batch = {}
            for future in futures:
                exception = future.exception()
                if exception:
                    logger.error(exception)
                    continue
                image = future.result()
                image_id = futures[future]
                if image is not None:
                    images_batch[image_id] = image
            return images_batch

    @classmethod
    def _get_masks_batch_data(cls, images_batch) -> dict:
        images_batch = dict(sorted(images_batch.items(), key=lambda x: x[0][0]))
        images_batch = itertools.groupby(images_batch.items(), key=lambda x: x[0][0])
        out_images = {}
        for group_id, group_images in images_batch:
            annot_df = pd.concat(
                [img_annot_df for img_id, img_annot_df in group_images], axis=0
            )
            files_to_extract = {row["MaskPath"] for _, row in annot_df.iterrows()}
            zip_file_to_extract_from = cls.get_mask_data_file_uri(group_id)
            extracted_files = FileUtils.extract_files_from_zip(
                files_to_extract, zip_file_to_extract_from
            )
            extracted_files = {
                fname: PILImage.open(io.BytesIO(fbytes))
                for fname, fbytes in extracted_files.items()
            }
            out_images = {**out_images, **extracted_files}
        return out_images

    @staticmethod
    def get_mask_data_file_uri(group_id):
        zip_file_uri = f"https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{group_id}.zip"
        return zip_file_uri

    @staticmethod
    def _fetch_image(split, image_id):
        try:
            bucket = "open-images-dataset"
            key = f"{split}/{image_id}.jpg"
            pil_image = S3Utils.fetch_image_unsigned(bucket, key)
            if pil_image is not None:
                return pil_image
        except Exception as ex:
            raise Exception(f"error downloading the image with id {image_id} : {ex}")
        return None
