import dataclasses
from dataclasses import dataclass


@dataclass
class PascalVOCAnnotation:
    xmin: int = 0
    ymin: int = 0
    xmax: int = 0
    ymax: int = 0
    score: float = 0.0
    name: str = ""
    scale: float = 1.0

    def __getitem__(self, item):
        return getattr(self, item)

    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def iou(self, other: "PascalVOCAnnotation") -> float:
        """
        Compute the intersection over union of two bounding boxes.
        """
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        w = xmax - xmin
        h = ymax - ymin
        if w < 0 or h < 0:
            return 0.0
        area = w * h
        return area / (self.area() + other.area() - area)

    def __post_init__(self):
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.type is int and isinstance(value, str):
                setattr(self, f.name, int(value))
