from abc import abstractmethod, ABCMeta
from concurrent.futures import ThreadPoolExecutor

from falconcv.util import ImageUtils
import logging

logger = logging.getLogger("rich")


class ImagesScraper(metaclass=ABCMeta):
    def __init__(self):
        self._image_files = []

    def _authenticate(self):
        pass

    @abstractmethod
    def _get_total_matches(self, query):
        pass

    @abstractmethod
    def _make_request(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch(self, *args, **kwargs):
        raise NotImplementedError

    def _download_images(self, files):
        with ThreadPoolExecutor(max_workers=10) as executor:
            images = []
            futures = {
                img_uri: executor.submit(ImageUtils.url2img, img_uri)
                for img_uri in files
            }
            for image_uri in futures:
                curr_future = futures[image_uri]
                future_exception = curr_future.exception()
                if future_exception:
                    logger.error(f"skipping image {image_uri} : {future_exception}")
                else:
                    images.append(curr_future.result())  # get result
        return images
