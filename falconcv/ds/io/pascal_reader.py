import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
from falconcv.util import Console


def read_pascal_dataset(xml_folder, images_folder=None):
    xml_list=[]
    files=glob.glob("{}/*.xml".format(xml_folder))
    for xml_file in files:
        images_folder=images_folder if images_folder else os.path.dirname(xml_file)
        tree=ET.parse(xml_file)
        root=tree.getroot()
        objects=root.findall('object')
        for member in objects:
            image_file_name=root.find('filename').text
            image_file_path=os.path.join(images_folder,image_file_name)
            if os.path.exists(image_file_path):
                value=(
                    xml_file,
                    image_file_path,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text))
                xml_list.append(value)
    column_names=['xml_path','image_path',"width",'height','class','xmin','ymin','xmax','ymax']
    return pd.DataFrame(xml_list,columns=column_names)
