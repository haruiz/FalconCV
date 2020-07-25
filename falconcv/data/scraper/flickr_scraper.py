import logging
import math
import re
import time
import dask
import numpy as np
import requests
import json
import xml.etree.ElementTree as ET
from falconcv.data.scraper.scraper import ImagesScraper
from falconcv.util import ImageUtil
logger = logging.getLogger(__name__)
FLICKR_ENDPOINT = "https://www.flickr.com/services/rest"
# List of sizes:
# url_o: Original (4520 × 3229)
# url_k: Large 2048 (2048 × 1463)
# url_h: Large 1600 (1600 × 1143)
# url_l=: Large 1024 (1024 × 732)
# url_c: Medium 800 (800 × 572)
# url_z: Medium 640 (640 × 457)
# url_m: Medium 500 (500 × 357)
# url_n: Small 320 (320 × 229)
# url_s: Small 240 (240 × 171)
# url_t: Thumbnail (100 × 71)
# url_q: Square 150 (150 × 150)
# url_sq: Square 75 (75 × 75)

class FlickrScraper(ImagesScraper):
    def __init__(self, api_key):
        super(FlickrScraper, self).__init__()
        self.api_key = api_key

    def _authenticate(self):
        pass

    def _get_total_matches(self, q):
        total_matches = 0
        try:
            response = requests.get(url=FLICKR_ENDPOINT, params={
                "api_key": self.api_key,
                "method": "flickr.photos.search",
                "tags": ",".join(q),
                "tag_mode": "any",
                # "privacy_filter": "1"
                "content_type": 1,
                "media": "photos",
                "per_page": 0,
                "format": "json"
            })
            if response.status_code == 200:
                json_text = re.search(r'\((.*?)\)', response.text).group(1)
                json_object = json.loads(json_text)
                if json_object["stat"] == "ok":
                    total_matches = int(json_object["photos"]["total"])
                    # total_matches = json_object["photos"]
        except Exception as ex:
            logger.error("Error making the request : {}".format(ex))
        return total_matches

    def _request_photos(self, q, count, page):
        images = []
        try:
            response = requests.get(url=FLICKR_ENDPOINT, params={
                "api_key": self.api_key,
                "method": "flickr.photos.search",
                "tags": ",".join(q),
                "tag_mode": "any",
                # "privacy_filter": "1"
                "content_type": 1,
                "media": "photos",
                "per_page": count,
                "page": page,
                "extras": ",".join(["url_o", "url_k", "url_h", "url_l", "url_c", "url_m"])
            })
            if response.status_code == 200:
                try:
                    # print(response.text)
                    root: ET.Element = ET.fromstring(response.text)
                    stat = root.get("stat")
                    if stat == "ok":
                        for photo in root.iterfind("photos/photo"):
                            photo: ET.Element
                            images.append(photo.attrib)
                except Exception as ex:
                    logger.error("error gathering the response: {}".format(ex))
        except Exception as ex:
            logger.error("Error making the request : {}".format(ex))
        return images

    @dask.delayed
    def _fetch_image(self, image_info, sz):
        try:
            if sz in image_info:
                url = image_info[sz]
                return ImageUtil.url2img(url)
        except Exception as ex:
            logger.error("Error fetching the image:  " % ex)
        return None

    def fetch(self, q, batch_size: int = 100, timestamp=1, sz="url_m"):
        try:
            assert batch_size <= 500, "invalid count parameter"
            total_matches = self._get_total_matches(q)
            logger.debug("{} images found ".format(total_matches))
            number_of_pages = math.ceil(total_matches / batch_size)
            for page in range(1, number_of_pages):
                photos = self._request_photos(q, batch_size, page)
                delayed_tasks = list(map(lambda img: self._fetch_image(img, sz), photos))
                compute_result = dask.compute(*delayed_tasks)
                yield [img for img in compute_result if isinstance(img, np.ndarray)]
                time.sleep(timestamp)
        except Exception as ex:
            logger.error("error fetching the images:  {}".format(ex))
