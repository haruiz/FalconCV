from dask.callbacks import Callback
from tqdm import tqdm


class ProgressBar(Callback):
    def _start_state(self, dsk, state):
        self._tqdm = tqdm(total=sum(len(state[k]) for k in ['ready', 'waiting', 'running', 'finished']))

    def _posttask(self, key, result, dsk, state, worker_id):
        self._tqdm.update(1)

    def _finish(self, dsk, state, errored):
        pass


class BoundingBox(object):
    def __init__(self,x1,y1,x2,y2,label,score=0,scale=1, mask=None):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.mask = mask
        self.scale=scale
        self.label=label
        self.score=score