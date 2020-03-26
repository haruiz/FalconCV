import itertools
import json
import os

import cv2
import dask
from falconcv.ds.image import TaggedImage,PolygonRegion,BoxRegion
from mako.template import Template
import numpy as np


class Encoder:
    def __init__(self,images):
        self._images = images

    def encode(self,output_folder):
        raise NotImplementedError

    def decode(self,input_file):
        raise NotImplementedError


class VGGEncoder(Encoder):
    def __init__(self,ds):
        super().__init__(ds)

    def encode(self,output_folder):
        data={}
        for tagged_image in self._images:
            file_name="{}.jpg".format(tagged_image.id)
            tagged_image.filename=file_name
            image_path=os.path.join(output_folder,file_name)
            img=cv2.cvtColor(tagged_image.img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path,img)
            tagged_image.size=os.path.getsize(image_path)
            data["{}{}".format(tagged_image.filename,tagged_image.size)]=tagged_image
        json_data=json.dumps(data,
                             default=lambda o: {k: v for k,v in o.__dict__.items() if k is not "img"},
                             sort_keys=True,
                             indent=4)
        os.makedirs(output_folder,exist_ok=True)
        with open(os.path.join(output_folder,"annotations.json"),'w') as json_file:
            json_file.write(json_data)


class PascalVOC(Encoder):
    def __init__(self,ds):
        super().__init__(ds)

    def encode(self,out_folder):

        def export_template(tagged_image: TaggedImage):
            str_template = '''
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
                        <name>${region.region_attributes["name"]}</name>
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
            os.makedirs(out_folder,exist_ok=True)
            image_file_name="{}.jpg".format(tagged_image.id)
            image_path =os.path.join(out_folder,image_file_name)
            img = tagged_image.img
            #img=cv2.cvtColor(tagged_image.img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path,img)
            if len(tagged_image.regions) > 0:
                polygons = [r for r in tagged_image.regions if isinstance(r, PolygonRegion)]
                boxes =[r for r in tagged_image.regions if isinstance(r,BoxRegion)]
                if len(polygons) > 0:
                    mask=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
                    for roi in polygons:
                        x = roi.shape_attributes["all_points_x"]
                        y = roi.shape_attributes["all_points_y"]
                        pts=np.array(list(zip(x,y)))
                        if pts.size > 0:
                            cv2.fillConvexPoly(mask,pts,255)
                    # res = cv2.bitwise_and(img,img,mask = mask)
                    mask_path = os.path.join(out_folder,"{}_mask.jpg".format(tagged_image.id))
                    cv2.imwrite(mask_path,mask)
                if len(boxes) > 0:
                    h,w = np.size(img, axis=0), np.size(img, axis=1)
                    c = np.shape(img)[-1] if len(np.shape(img)) == 3 else 1
                    xml_str = Template(str_template).render(
                        path=image_path,
                        folder=out_folder,
                        filename= image_file_name,
                        width=w,
                        height=h,
                        depth=c,
                        annotations=boxes)
                    image_file_name,_=os.path.splitext(image_file_name)
                    output_file=os.path.join(out_folder,"{}.xml".format(image_file_name))
                    with open(output_file,'w') as f:
                        f.write(xml_str)

        dask.compute(*[dask.delayed(export_template)(img) for img in self._images])
