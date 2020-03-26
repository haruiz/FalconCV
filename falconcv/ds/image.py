import numpy as np


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
