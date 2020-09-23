import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from falconcv.util import FileUtil


class DtPascalVOCDataset(object):
    def __init__(self, class_names: list):
        self._class_names = class_names

    def register(self, name: str, images_folder: Path, xml_folder: Path, split: str):
        DatasetCatalog.register(name, lambda: self._get_dataset_dicts(images_folder, xml_folder))
        MetadataCatalog.get(name).set(thing_classes=self._class_names,
                                      dirname=str(images_folder),
                                      split=split)

    def _get_dataset_dicts(self, images_folder, xml_folder):
        img_files = FileUtil.get_files(images_folder, [".jpg", ".jpeg"])
        xml_files = FileUtil.get_files(xml_folder, [".xml"])

        files = img_files + xml_files
        files = sorted(files, key=lambda img: img.stem)

        dicts = []
        for img_name, img_files in itertools.groupby(files, key=lambda img: img.stem):
            img_file, xml_file = None, None
            for file in img_files:
                if file.suffix in [".jpg", ".jpeg"]:
                    img_file = file
                elif file.suffix == ".xml":
                    xml_file = file
            if img_file and xml_file:
                dicts.append(self._get_record(img_file, xml_file))

        return dicts

    def _get_record(self, image_file, xml_file):
        tree = ET.parse(xml_file)
        record = {
            "file_name": image_file,
            "image_id": image_file,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            instances.append(
                {"category_id": self._class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        record["annotations"] = instances

        return record
