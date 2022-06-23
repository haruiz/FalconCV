import io

import numpy as np
from PIL import Image
from lxml import etree


class PascalVOCUtils:
    @classmethod
    def recursive_parse_xml_to_dict(cls, xml):
        if xml is None or len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = cls.recursive_parse_xml_to_dict(child)
            if child.tag != "object":
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    @classmethod
    def load_annotations(cls, xml_path):
        from falconcv.data.pascalvoc_annotation import PascalVOCAnnotation

        with open(xml_path, "r") as f:
            xml_content = f.read()
            xml_tree = etree.fromstring(xml_content)
        anno_dict = cls.recursive_parse_xml_to_dict(xml_tree)["annotation"]
        annotations = []
        for bbox_dict in anno_dict["object"]:
            annotations.append(
                PascalVOCAnnotation(
                    name=bbox_dict["name"].lower(), **bbox_dict["bndbox"]
                )
            )
        return annotations

    @staticmethod
    def load_image(image_path, for_tf_record=False):
        image_path = str(image_path)
        with open(image_path, "rb") as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        pil_image = Image.open(encoded_jpg_io)
        if pil_image.format not in ["JPEG", "MPO"]:
            raise ValueError("Image image format not JPG")
        img_arr = np.asarray(pil_image)
        if for_tf_record:
            return img_arr, encoded_jpg
        return img_arr

    @staticmethod
    def load_mask(mask_path, for_tf_record=False):
        mask_path = str(mask_path)
        with open(mask_path, "rb") as fid:
            encoded_mask_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_mask_png)
        pil_mask = Image.open(encoded_png_io)
        if pil_mask.format != "PNG":
            raise ValueError("Mask image format not PNG")
        mask_arr = np.asarray(pil_mask)
        if for_tf_record:
            return mask_arr, encoded_mask_png
        return mask_arr
