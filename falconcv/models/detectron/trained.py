import numpy as np
import logging
import abc

from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from falconcv.models import ApiModel, ModelConfig
from falconcv.decor import typeassert
from falconcv.util.img_util import ImageUtil
from .util import Utilities

logger = logging.getLogger(__name__)
setup_logger()


class DtTrainedModel(ApiModel):
    @abc.abstractmethod
    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config
        self._dt_config = None
        self._predictor = None

    def __enter__(self):
        self.load_labels_map()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Error loading the model:  {}, {}".format(exc_type, str(exc_val)))

    def load_labels_map(self):
        pass

    @abc.abstractmethod
    @typeassert(input_image=np.ndarray)
    def output(self, input_image: np.ndarray):
        raise NotImplementedError()

    @typeassert(input_image=[str, np.ndarray], size=tuple)
    def predict(self, input_image, size=None, threshold=0.5, top_k=10):
        print("[INFO] pre-processing image...")
        img_arr, img_width, img_height, scale_factor = ImageUtil.process_input_image(input_image, size)

        print("[INFO] making prediction...")
        self._dt_config = Utilities.update_config(self._dt_config, threshold, top_k)
        if self._predictor is None:
            self._predictor = DefaultPredictor(self._dt_config)
        predictions = self.output(img_arr)

        print("[INFO] making annotations...")
        vis_image = Visualizer(img_arr[:, :, ::-1], MetadataCatalog.get(
            self._dt_config.DATASETS.TRAIN[0]), scale=1.2)
        vis_image = vis_image.draw_instance_predictions(predictions["instances"].to("cpu"))
        return vis_image.get_image()


class DtFreezeModel(DtTrainedModel):
    @typeassert(model_config=ModelConfig)
    def __init__(self, model_config: ModelConfig):
        print("[INFO] detectron2 predictions...")
        super(DtFreezeModel, self).__init__(model_config)

    def __enter__(self):
        print("[INFO] loading pre-trained model...")
        super(DtFreezeModel, self).__enter__()
        # self._dt_config, self._predictor = Utilities.load_predictor(self._model_config.model)
        self._dt_config = Utilities.load_config(self._model_config.model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(DtFreezeModel, self).__exit__(exc_type, exc_val, exc_tb)

    def output(self, img_arr: np.ndarray):
        if self._dt_config and self._predictor:
            predictions = self._predictor(img_arr)
            return predictions
        return None
