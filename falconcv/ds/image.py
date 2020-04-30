from pathlib import Path

from PIL import Image, ImageDraw
import cv2
import numpy as np
from mako.template import Template

pascal_voc_template = '''
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
            <name>${region.region_attributes["name"].title()}</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>${region.shape_attributes["x"]}</xmin>
                <ymin>${region.shape_attributes["y"]}</ymin>
                <xmax>${region.shape_attributes["x"] + region.shape_attributes["width"]}</xmax>
                <ymax>${region.shape_attributes["y"] + region.shape_attributes["height"]}</ymax>
            </bndbox>
        </object>
    % endfor
</annotation>
'''


class BoxRegion:
    def __init__(self):
        self.shape_attributes={
            "name": "rect",
            "x": None,
            "y": None,
            "width": None,
            "height": None
        }
        self.region_attributes={
            "name": None,
            "is_occluded": 0,
            "is_truncated": 0,
            "is_group_of": 0,
            "is_depiction": 0,
            "is_inside": 0
        }


class PolygonRegion:
    def __init__(self):
        self.shape_attributes={
            "name": "polygon",
            "all_points_x": [],
            "all_points_y": [],
            "x": None,
            "y": None,
            "width": None,
            "height": None
        }
        self.region_attributes={
            "name": None
        }


class TaggedImage:
    def __init__(self,img: np.ndarray):
        self.img: np.ndarray=img
        self.id=""
        self.filename=""
        self.size=0
        self.uri: str=""
        self.regions: []=[]
        self.file_attributes: dict={}

    def __dir__(self):
        return super().__dir__()

    def export(self, out_folder: str, labels_map=None, color_palette= None):
        out_folder = Path(out_folder) if isinstance(out_folder, str) else out_folder
        out_folder.mkdir(exist_ok=True)
        # create xml
        self.filename= "{}.jpeg".format(self.id)
        image = Image.fromarray(self.img).convert("RGB")
        width, height = image.size
        number_of_channels = len(image.getbands())
        # export image
        xml_str = Template(pascal_voc_template)\
            .render(
            path=out_folder.joinpath(self.filename),
            folder=out_folder,
            filename=self.filename,
            width=width,
            height=height,
            depth=number_of_channels,
            annotations=self.regions)
        xml_file = out_folder.joinpath("{}.xml".format(self.id))
        with open(str(xml_file), 'w') as f:
            f.write(xml_str)
        # export image
        image.save(str(out_folder.joinpath(self.filename))) # export image
        # export mask
        polygons = [r for r in self.regions if isinstance(r, PolygonRegion)]
        if len(polygons) > 0:
            mask = Image.new("P", (width, height), 0)
            mask.putpalette(color_palette)
            for region in polygons:
                if isinstance(region, PolygonRegion):
                    x = region.shape_attributes["all_points_x"]
                    y = region.shape_attributes["all_points_y"]
                    pts = list(zip(x, y))
                    if len(pts) > 0:
                        if labels_map:
                            class_name = region.region_attributes["name"]
                            if class_name in labels_map:
                                drawable_image = ImageDraw.Draw(mask)
                                label_id = labels_map[class_name]
                                drawable_image.polygon(pts, fill=label_id)
                                #drawable_image.polygon(pts, fill=colors_map[class_name],outline="white")
                                del drawable_image
                        else:
                            #cv2.fillConvexPoly(mask, pts, 255)
                            drawable_image = ImageDraw.Draw(mask)
                            drawable_image.polygon(pts, fill="white")
                            del drawable_image
            # res = cv2.bitwise_and(img,img,mask = mask)
            mask.save(str(out_folder.joinpath("{}.png".format(self.id))), "PNG")  # export image
