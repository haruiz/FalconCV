
class BoundingBox(object):
    def __init__(self,x1,y1,x2,y2,label,score,scale=1, mask=None):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.mask = mask
        self.scale=scale
        self.label=label
        self.score=score
