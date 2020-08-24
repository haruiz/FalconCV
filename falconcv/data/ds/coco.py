import logging
import math
import random

import dask
from pycocotools.coco import COCO
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from falconcv.util import FileUtil, ImageUtil
from .image import TaggedImage, BoxRegion, PolygonRegion
from .remote_dataset import RemoteDataset

logger = logging.getLogger(__name__)


class Coco(RemoteDataset):

    def __init__(self, version: int = 2017, task="detection", split: str = "train", labels: [str] = None,
                 n_images: int = 0,
                 batch_size: int = 12):
        """
        Create an instance of the coco dataset
        :param v: version of the dataset
        :param labels: labels to download
        :param count: number of images by label
        :param batch_size: number of images by batch
        """
        assert version == 2017, "version not supported"
        assert task in ["detection", "segmentation"], "task not supported"
        assert split in ["train", "validation"], "invalid split parameter"
        super(Coco, self).__init__(version, split, labels, n_images, batch_size)
        self._task = task
        self._files = {
            "annotations_file_uri": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        }
        self.__load__()

    def __load__(self):
        """
        Download dataset dependencies
        """
        try:
            self._download_dependencies()
            ann_folder = self.home().joinpath("annotations")
            ann_file_prefix = "train" if self._split == "train" else "val"
            ann_file = ann_folder.joinpath(f"instances_{ann_file_prefix}{self._version}.json")
            self._coco_api_client = COCO(str(ann_file))
            cat_ids = self._coco_api_client.getCatIds()
            cats_info = self._coco_api_client.loadCats(cat_ids)
            self._available_labels = {cat['id']: cat['name'].capitalize() for cat in cats_info}
            # capitalize labels
            labels = list(map(lambda l: str(l).capitalize(), self._labels))
            valid_labels = {id: name for id, name in self._available_labels.items() if name.capitalize() in labels}
            for class_id, class_name in valid_labels.items():
                images_ids = self._coco_api_client.getImgIds(catIds=class_id)
                # grab the n images available
                if self._n_images:
                    count = min(self._n_images, len(images_ids))
                    sample_images = random.sample(images_ids, count)
                    self._images += zip(sample_images, [class_id] * len(sample_images))
        except Exception as ex:
            logger.error("Error loading the dataset : {} ".format(ex))
            raise ex

    def __getitem__(self, item):
        return super(Coco, self).__getitem__(item)

    @staticmethod
    def _create_box_rois(annotations, class_name):
        regions = []
        try:
            for ann_info in annotations:
                r = BoxRegion()
                bb = ann_info['bbox']
                r.shape_attributes["x"] = math.ceil(bb[0])
                r.shape_attributes["y"] = math.ceil(bb[1])
                r.shape_attributes["width"] = math.ceil(bb[2])
                r.shape_attributes["height"] = math.ceil(bb[3])
                r.region_attributes["name"] = class_name
                regions.append(r)
        except Exception as ex:
            logger.error(f"error processing annotations : {ex}")
        return regions

    @staticmethod
    def _create_polygon_rois(annotations, class_name):
        regions = []
        try:
            for ann_info in annotations:
                polygons = ann_info['segmentation'] or []
                for polygon in polygons:
                    all_x, all_y = [], []
                    for i in range(0, len(polygon), 2):
                        try:
                            a, b = polygon[i], polygon[i + 1]
                            if all(map(lambda pt: isinstance(pt, float), [a, b])):
                                all_x.append(math.ceil(polygon[i]))
                                all_y.append(math.ceil(polygon[i + 1]))
                        except:
                            continue
                        if all_x and all_y:
                            bb = ann_info['bbox']
                            r = PolygonRegion()
                            r.shape_attributes["all_points_x"] = all_x
                            r.shape_attributes["all_points_y"] = all_y
                            r.shape_attributes["x"] = math.ceil(bb[0])
                            r.shape_attributes["y"] = math.ceil(bb[1])
                            r.shape_attributes["width"] = math.ceil(bb[2])
                            r.shape_attributes["height"] = math.ceil(bb[3])
                            r.region_attributes["name"] = class_name
                            regions.append(r)
        except Exception as ex:
            logger.error(f"error processing annotations : {ex}")
        return regions

    def _fetch_image(self, img_info, category_id):
        """
        load the image into memory and get the annotations
        :param img_info: coco image metadata
        :param category_id: category id
        :return:
        """
        try:
            img_uri = img_info["coco_url"]
            img_id = img_info['id']
            class_name = self._available_labels[category_id]
            annotations_ids = self._coco_api_client.getAnnIds(imgIds=img_id, catIds=category_id)
            annotations_info = self._coco_api_client.loadAnns(ids=annotations_ids)
            if FileUtil.exists_http_file(img_uri):
                img_arr = ImageUtil.url2img(img_uri)
                tagged_image = TaggedImage(img_arr)
                tagged_image.id = img_id
                if self._task == "detection":
                    tagged_image.regions = self._create_box_rois(annotations_info, class_name)
                elif self._task == "segmentation":
                    tagged_image.regions = self._create_polygon_rois(annotations_info, class_name)
                return tagged_image
        except Exception as ex:
            logger.error("error downloading the image with id {} : {}".format(img_info["id"], ex))
        return None

    def __getbatch__(self, batch):
        images_ids = [img_id for img_id, _ in batch]
        categories_ids = [category_id for _, category_id in batch]
        images_details = self._coco_api_client.loadImgs(ids=images_ids)
        # results = []
        # for img_info, category_id in tqdm(zip(images_details, categories_ids)):
        #     img = self._fetch_image(img_info, category_id)
        #     if img:
        #         results.append(img)
        # return results
        delayed_tasks = []
        for img_info, category_id in zip(images_details, categories_ids):
            delayed_tasks.append(
                dask.delayed(
                    self._fetch_image(
                        img_info,
                        category_id
                    )
                )
            )
        with ProgressBar():
            results = dask.compute(*delayed_tasks)
        results = [img for img in results if isinstance(img, TaggedImage)]
        return results  # download the images in parallel using dask
