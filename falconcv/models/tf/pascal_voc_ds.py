import hashlib
import io
from pathlib import Path

import dask
import more_itertools
import tensorflow as tf
from PIL import Image
from lxml import etree
from object_detection.utils import dataset_util
from object_detection.utils.dataset_util import bytes_list_feature, int64_list_feature, float_list_feature, \
    bytes_feature, int64_feature
import numpy as np
import logging

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class PascalVOCImage:
    def __init__(self, img_path, xml_Path, mask_Path):
        self.img_path: Path = img_path
        self.xml_path: Path = xml_Path
        self.mask_path: Path = mask_Path
        self.annotations = {}

    def load(self):
        try:
            if self.xml_path.exists():
                with tf.io.gfile.GFile(str(self.xml_path), 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                self.annotations = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        except Exception as ex:
            print(f"Error reading the file {ex}: {self.xml_path}")
        

    def to_example_record(self, labels_map):
        # read image

        try:
            with tf.io.gfile.GFile(str(self.img_path), 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image: Image = Image.open(encoded_jpg_io)
            width, height = image.size
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            image_key = hashlib.sha256(encoded_jpg).hexdigest()
        except Exception as ex:
            logger.warn("Error encoding image {}, image ignored".format(self.img_path))
            return None

        # read mask
        mask = None
        try:
            if self.mask_path and self.mask_path.exists():
                with tf.io.gfile.GFile(str(self.mask_path), 'rb') as fid:
                    encoded_mask_png = fid.read()
                encoded_png_io = io.BytesIO(encoded_mask_png)
                mask = Image.open(encoded_png_io)
                width, height = mask.size
                if mask.format != 'PNG':
                    raise ValueError('Mask image format not PNG')
                mask = np.asarray(mask)
        except Exception as ex:
            logger.warn("Error encoding the mask file {}, image ignored".format(self.img_path))
            return None

        # create records
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        classes, classes_text, encoded_masks = [], [], []
        encoded_mask_png_list = []
        if self.annotations:
            if 'object' in self.annotations:
                
                for obj in self.annotations['object']:
                    class_name = obj['name'].strip().title()
                    class_id = labels_map[class_name]
                    xmin = float(obj['bndbox']['xmin'])
                    xmax = float(obj['bndbox']['xmax'])
                    ymin = float(obj['bndbox']['ymin'])
                    ymax = float(obj['bndbox']['ymax'])
                    xmins.append(xmin / width)
                    ymins.append(ymin / height)
                    xmaxs.append(xmax / width)
                    ymaxs.append(ymax / height)
                    classes_text.append(class_name.encode('utf8'))
                    classes.append(class_id)
                    # if a mask exist
                    if isinstance(mask, np.ndarray):
                        # object mask
                        mask_roi = np.zeros_like(mask)
                        mask_roi[int(ymin):int(ymax), int(xmin):int(xmax)] = mask[int(ymin):int(ymax),
                                                                             int(xmin):int(xmax)]
                        mask_remapped = (mask_roi == class_id).astype(np.uint8)
                        mask_remapped = Image.fromarray(mask_remapped)
                        output = io.BytesIO()
                        mask_remapped.save(output, format='PNG')
                        encoded_mask_png_list.append(output.getvalue())

            feature_dict = {
                'image/height': int64_feature(height),
                'image/width': int64_feature(width),
                'image/filename': bytes_feature(self.img_path.name.encode('utf8')),
                'image/source_id': bytes_feature(self.img_path.name.encode('utf8')),
                'image/key/sha256': bytes_feature(image_key.encode('utf8')),
                'image/encoded': bytes_feature(encoded_jpg),
                'image/format': bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': float_list_feature(xmins),
                'image/object/bbox/xmax': float_list_feature(xmaxs),
                'image/object/bbox/ymin': float_list_feature(ymins),
                'image/object/bbox/ymax': float_list_feature(ymaxs),
                'image/object/class/text': bytes_list_feature(classes_text),
                'image/object/class/label': int64_list_feature(classes)}
            if len(encoded_mask_png_list) > 0:
                feature_dict['image/object/mask'] = bytes_list_feature(encoded_mask_png_list)
            return tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return None


def PascalVOCGenerator(images, batch_size = 200):
    for i,batch_of_images in enumerate(more_itertools.chunked(images,batch_size)):
        yield batch_of_images


class PascalVOCDataset:
    def __init__(self, ):
        self._images = []

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, value):
        self._images = value

    def load(self, images: list, batch_size: int = 200):
        self._images = images
        for batch_of_images in PascalVOCGenerator(self._images, batch_size):
            delayed_task = [dask.delayed(img.load)() for img in batch_of_images]
            dask.compute(*delayed_task)

    @property
    def count(self):
        return len(self._images)

    def __iter__(self):
        return iter(self._images)

    @property
    def empty(self):
        return len(self._images) == 0

    def split(self, split_size=float):
        return train_test_split(self._images, test_size=split_size, random_state=42, shuffle=True)


