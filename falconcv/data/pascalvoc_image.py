import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mako.template import Template

from falconcv.decor import depends, exception
from .pascalvoc_annotation import PascalVOCAnnotation
from falconcv.util import TODAUtils, ColorUtils, VisUtils
from .pascalvoc_utils import PascalVOCUtils
from PIL import Image as PILImage
import io
import matplotlib.pyplot as plt
from typing_extensions import get_args

try:
    import tensorflow.compat.v1 as tf
    from object_detection.utils.label_map_util import (
        create_category_index_from_labelmap,
    )
except ImportError:
    ...

pascalvoc_template = """
<annotation>
   <folder>${folder}</folder>
   <filename>${filename}</filename>
   <path>${path}</path>
   <source>
       <database>Unknown</database>
   </source>
   <size>
       <width>${width}</width>
       <height>${height}</height>
       <depth>${depth}</depth>
   </size>
   <segmented>0</segmented>
   % for i, region in enumerate(annotations):
       <object>
           <name>${region["name"]}</name>
           <pose>Unspecified</pose>
           <truncated>0</truncated>
           <difficult>0</difficult>
           <bndbox>
               <xmin>${region["xmin"]}</xmin>
               <ymin>${region["ymin"]}</ymin>
               <xmax>${region["xmax"]}</xmax>
               <ymax>${region["ymax"]}</ymax>
           </bndbox>
       </object>
   % endfor
</annotation>
"""


@dataclass
class PascalVOCImage:
    image_path: typing.Union[Path, str] = None
    mask_path: typing.Union[Path, str] = None
    xml_path: typing.Union[Path, str] = None
    image_arr: np.ndarray = None
    mask_arr: np.ndarray = None
    _annotations: typing.List[PascalVOCAnnotation] = field(default_factory=lambda: [])

    @property
    def image(self):
        if isinstance(self.image_arr, (np.ndarray, PILImage.Image)):
            return self.image_arr
        elif self.image_path is not None and self.image_path.exists():
            self.image_arr = PascalVOCUtils.load_image(self.image_path)
        return self.image_arr

    @image.setter
    def image(self, image):
        self.image_arr = image

    @property
    def mask(self):
        if isinstance(self.mask_arr, (np.ndarray, PILImage.Image)):
            return self.mask_arr
        elif self.mask_path is not None and self.mask_path.exists():
            self.mask_arr = PascalVOCUtils.load_mask(self.mask_path)
        return self.mask_arr

    @mask.setter
    def mask(self, mask):
        self.mask_arr = mask

    # lazy load
    @property
    def annotations(self):
        if self._annotations:
            return self._annotations
        elif self.xml_path is not None and self.xml_path.exists():
            self._annotations = PascalVOCUtils.load_annotations(self.xml_path)
        return self._annotations

    @annotations.setter
    def annotations(self, value):
        self._annotations = value

    def __post_init__(self):
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if Path in get_args(f.type) and value is not None:
                setattr(self, f.name, Path(value))

    def __str__(self):
        assert self.image_path is not None, "image_path not set"
        c, h, w = self.shape()
        return Template(pascalvoc_template).render(
            path=self.image_path.absolute(),
            folder=self.image_path.absolute().parent.name,
            filename=self.image_path.name,
            width=w,
            height=h,
            depth=c,
            annotations=self.annotations,
        )

    def shape(self):
        img_arr = self.image
        if isinstance(img_arr, np.ndarray):
            h, w = img_arr.shape[:2]
            if len(img_arr.shape) == 3:
                c = img_arr.shape[2]
            else:
                c = 1
        else:
            w, h = img_arr.size
            c = len(img_arr.getbands())
        return c, h, w

    @depends("object_detection")
    def to_example_train(self, labels_map):
        assert (
            self.image_path is not None and self.image_path.exists()
        ), "image_path does not exist or not set"
        img_arr = self.image
        mask_arr = self.mask

        pil_image = (
            PILImage.fromarray(img_arr) if isinstance(img_arr, np.ndarray) else img_arr
        )
        pil_image = pil_image.convert("RGB")
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        width, height = pil_image.size

        annotations = self.annotations
        labels = [labels_map[ann.name] for ann in annotations]
        feature_dict = {
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[img_bytes])
            ),
            "image/format": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=["jpeg".encode("utf-8")])
            ),
            "image/height": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[height])
            ),
            "image/width": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[width])
            ),
            "image/filename": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[self.image_path.name.encode("utf-8")]
                )
            ),
            "image/source_id": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[self.image_path.name.encode("utf-8")]
                )
            ),
            "image/object/bbox/xmin": tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=[float(ann.xmin) / float(width) for ann in annotations]
                )
            ),
            "image/object/bbox/ymin": tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=[float(ann.ymin) / float(height) for ann in annotations]
                )
            ),
            "image/object/bbox/xmax": tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=[float(ann.xmax) / float(width) for ann in annotations]
                )
            ),
            "image/object/bbox/ymax": tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=[float(ann.ymax) / float(height) for ann in annotations]
                )
            ),
            "image/object/class/label": tf.train.Feature(
                int64_list=tf.train.Int64List(value=labels)
            ),
            "image/object/class/text": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[ann.name.encode("utf-8") for ann in annotations]
                )
            ),
        }

        if isinstance(self.mask_arr, (np.ndarray, PILImage.Image)):
            if isinstance(self.mask_arr, PILImage.Image):
                mask_arr = np.array(self.mask_arr)
            encoded_mask_png_list = []
            for ann in annotations:
                ann_mask = np.zeros_like(mask_arr)
                ann_mask[ann.ymin : ann.ymax, ann.xmin : ann.xmax] = mask_arr[
                    ann.ymin : ann.ymax, ann.xmin : ann.xmax
                ]
                mask_remapped = (ann_mask == labels_map[ann.name]).astype(np.uint8)
                mask_remapped = PILImage.fromarray(mask_remapped)
                output = io.BytesIO()
                mask_remapped.save(output, format="PNG")
                encoded_mask_png_list.append(output.getvalue())
            feature_dict["image/object/mask"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=encoded_mask_png_list)
            )
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    @exception
    def plot(
        self,
        labels_map=None,
        labels_colors_map=None,
        figsize=None,
        box_opacity=0.50,
        mask_opacity=0.2,
        fontsize=12,
        include_labels=True,
        matplotlib_backend="tkagg",
    ):
        with VisUtils.with_matplotlib_backend(matplotlib_backend):
            annotations = self.annotations
            image_arr = self.image
            mask_arr = self.mask

            if isinstance(mask_arr, PILImage.Image):
                mask_arr = np.array(mask_arr)
            if isinstance(image_arr, PILImage.Image):
                image_arr = np.array(image_arr)

            if labels_map is None:
                labels_map = {ann.name: i + 1 for i, ann in enumerate(annotations)}

            if labels_colors_map is None:
                labels_colors_map = ColorUtils.labels_colors_map(
                    list(labels_map.keys()), box_opacity
                )

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, aspect="equal")

            TODAUtils.draw_boxes(annotations, labels_colors_map, ax)

            if isinstance(mask_arr, np.ndarray) and labels_map:
                TODAUtils.draw_mask(
                    image_arr, mask_arr, labels_map, labels_colors_map, mask_opacity
                )

            if include_labels:
                TODAUtils.draw_labels(annotations, labels_colors_map, ax, fontsize)

            ax.set_axis_off()
            ax.imshow(image_arr)
            plt.tight_layout()
            plt.show()
            return fig

    @exception
    def save(self, output_path: typing.Union[Path, str], color_palette=None):
        output_path = Path(output_path)
        assert output_path.suffix == ".jpg", "invalid file extension"
        self.image_path = output_path

        self.xml_path = self.image_path.with_suffix(".xml")
        with self.xml_path.open("w") as f:
            f.write(str(self))

        if isinstance(self.image_arr, np.ndarray):
            pil_image = PILImage.fromarray(self.image_arr).convert("RGB")
            pil_image.save(self.image_path)
        else:
            self.image_arr.save(self.image_path)

        self.mask_path = self.image_path.with_suffix(".png")
        if isinstance(self.mask_arr, np.ndarray):
            pil_image = PILImage.fromarray(self.mask_arr)
            if color_palette is not None:
                pil_image.putpalette(color_palette)
            pil_image.save(self.mask_path)
        elif isinstance(self.mask_arr, PILImage.Image):
            self.mask_arr.save(self.mask_path)

    def add_box(self, name, xmin, ymin, xmax, ymax):
        self.annotations.append(
            PascalVOCAnnotation(
                name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, scale=1
            )
        )
