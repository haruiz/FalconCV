import math
import os
import logging
import random
from pathlib import Path

import dask
import more_itertools
from pycocotools.coco import COCO

from falconcv.ds.image import TaggedImage,BoxRegion,PolygonRegion
from falconcv.util import FileUtil,ImageUtil

logger=logging.getLogger(__name__)
from falconcv.ds.dataset import DatasetDownloader


class Coco(DatasetDownloader):
    def __init__(self, v = 2017):
        super(Coco, self).__init__()
        assert v == 2017, "version not supported"
        config_path,_=os.path.splitext(__file__)
        self._annotations_json_file = None
        self._v = v
        self._coco_api_client = None
        self._remote_dep = {
            "annotations_file_uri": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        }

    def _create_box_rois(self,image_info,class_name):
        regions=[]
        try:
            category_id=self.labels_map[class_name]
            annotations_ids=self._coco_api_client.getAnnIds(imgIds=image_info['id'],catIds=category_id)
            annotations_info=self._coco_api_client.loadAnns(ids=annotations_ids)
            for ann_info in annotations_info:
                r=BoxRegion()
                bb=ann_info['bbox']
                r.shape_attributes["x"]=math.ceil(bb[0])
                r.shape_attributes["y"]=math.ceil(bb[1])
                r.shape_attributes["width"]=math.ceil(bb[2])
                r.shape_attributes["height"]=math.ceil(bb[3])
                r.region_attributes["name"]=class_name
                regions.append(r)
        except Exception as ex:
            print(ex)
        return regions

    def _create_polygon_rois(self,image_info,class_name):
        regions=[]
        category_id=self.labels_map[class_name]
        annotations_ids=self._coco_api_client.getAnnIds(imgIds=image_info['id'],catIds=category_id)
        annotations_info=self._coco_api_client.loadAnns(ids=annotations_ids)
        for ann_info in annotations_info:
            polygons =ann_info['segmentation']
            if len(polygons) > 0:
                for polygon in polygons:
                    all_x, all_y=[], []
                    for i in range(0,len(polygon),2):
                        try:
                            if isinstance(polygon[i], float) and \
                               isinstance(polygon[i+1],float):
                                    all_x.append(math.ceil(polygon[i]))
                                    all_y.append(math.ceil(polygon[i+1]))
                        except Exception:
                            pass
                    if len(all_x) > 0 and len(all_y) > 0:
                        bb = ann_info['bbox']
                        r=PolygonRegion()
                        r.shape_attributes["all_points_x"]=all_x
                        r.shape_attributes["all_points_y"]=all_y
                        r.shape_attributes["x"] = math.ceil(bb[0])
                        r.shape_attributes["y"] = math.ceil(bb[1])
                        r.shape_attributes["width"] = math.ceil(bb[2])
                        r.shape_attributes["height"] = math.ceil(bb[3])
                        r.region_attributes["name"] = class_name
                        regions.append(r)
        return regions

    @dask.delayed
    def _fetch_single_image(self, img_info, image_id, image_label):
        try:
            img_uri = img_info["coco_url"]
            if FileUtil.exists_http_file(img_uri):
                img_arr = ImageUtil.url2img(img_uri)
                tagged_image=TaggedImage(img_arr)
                tagged_image.id=image_id
                if self.task == "detection":
                    tagged_image.regions = self._create_box_rois(img_info, image_label)
                elif self.task == "segmentation":
                    tagged_image.regions=self._create_polygon_rois(img_info,image_label)
                return tagged_image
        except Exception as ex:
            print(ex)
            logger.error("error downloading the image with id {} : {}".format(image_id,ex))
        return None

    def fetch(self,n=None,labels=None,batch_size: int = 200):
        try:
            assert self._coco_api_client, "did you forget to call the setup method?"
            labels =list(map(lambda l: l.capitalize(),labels))
            valid_labels= {name: id for name, id in self.labels_map.items() if name.capitalize() in labels}
            for class_name, class_id in valid_labels.items():
                logger.info("downloading images for : {}".format(class_name))
                images_ids=self._coco_api_client.getImgIds(catIds=class_id)
                if n:
                    count=min(n,len(images_ids))
                    images_ids=random.sample(images_ids,count)
                number_of_batches = math.ceil(len(images_ids) / batch_size)
                for i,batch_ids in enumerate(more_itertools.chunked(images_ids,batch_size)):
                    images_batch=self._coco_api_client.loadImgs(ids=batch_ids)
                    delayed_tasks=[]
                    for img_info, image_id in zip(images_batch, batch_ids):
                        delayed_tasks.append(
                            self._fetch_single_image(
                                img_info,
                                image_id,
                                class_name
                            )
                        )
                    logger.info("downloading batch {}/{}".format(i+1,number_of_batches))
                    if len(delayed_tasks) > 0:
                        results=dask.compute(*delayed_tasks)
                        results=[img for img in results if img]
                        yield results
                        del results
        except Exception as ex:
            logger.exception("Error fetching the images : {} ".format(ex))  # in case something wrong happens
            raise ex

    def setup(self, split="train", task="detection"):
        try:
            assert task == "detection" or task=="segmentation","task not supported"
            assert split in ["train","validation"],"invalid split parameter"
            super(Coco, self).setup(split, task)
            ann_zip_file: Path =self._dependencies["annotations_file_uri"]
            ann_folder=ann_zip_file.parent.joinpath("annotations")
            ann_file_prefix="train" if split == "train" else "val"
            ann_file=ann_folder.joinpath("instances_{}{}.json".format(ann_file_prefix, self._v))
            self._coco_api_client=COCO(str(ann_file))
            cat_ids=self._coco_api_client.getCatIds()
            cats_info=self._coco_api_client.loadCats(cat_ids)
            self.labels_map={cat['name']:cat['id'] for cat in cats_info}
            self.slabels_map = self.labels_map
        except Exception as ex:
            logger.error("Error preparing the dataset : {} ".format(ex))
            raise ex


