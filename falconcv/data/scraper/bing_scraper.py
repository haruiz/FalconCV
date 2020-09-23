import time

import dask
import more_itertools

from falconcv.data.scraper.scraper import ImagesScraper
from falconcv.util import FileUtil,ImageUtil
from requests import exceptions
from urllib3.exceptions import ProtocolError
import requests
import numpy as np
import logging
logger=logging.getLogger(__name__)

BING_IMAGE_SEARCH_ENDPOINT="https://api.cognitive.microsoft.com/bing/v7.0/images/search"
EXCEPTIONS={ProtocolError,IOError,FileNotFoundError,exceptions.RequestException,exceptions.HTTPError,exceptions.ConnectionError, exceptions.Timeout}


class BingScraper(ImagesScraper):
    def __init__(self,subscription_key,endpoint=None):
        super(BingScraper, self).__init__()
        self.subscription_key=subscription_key
        self.endpoint=endpoint if endpoint else BING_IMAGE_SEARCH_ENDPOINT

    def _get_total_matches(self, q):
        total_matches = 0
        resp=requests.get(
            self.endpoint,
            params={
                "q": q,
                "imageType": "Photo",
                "license": "All",
                "size": "Large"
            },
            headers={
                "Ocp-Apim-Subscription-Key": self.subscription_key
            })
        if resp.status_code == 200:
            total_matches =  resp.json()["totalEstimatedMatches"]
        elif resp.status_code in [401, 410, 429]:
            raise Exception(resp.json()["error"]["message"])
        return total_matches

    def _make_api_call(self,  q, offset,  count):
        images  = []
        resp=requests.get(
          self.endpoint,
          params={
              "q": q,
              "imageType": "Photo",
              "license": "All",
              "size": "Large",
              "count": count,
              "offset": offset
          },
          headers={
              "Ocp-Apim-Subscription-Key": self.subscription_key
          })
        if resp.status_code == 200:
            resp=resp.json()
            for img in resp["value"]:
                img_uri=img["contentUrl"]
                images.append(img_uri)
        elif resp.status_code in [401,410,429]:
            raise Exception(resp.json()["error"]["message"])
        return images

    def fetch(self,q,count=100, batch_size: int = 200, timestamp = 1):
        try:
            total_matches = self._get_total_matches(q)
            logger.debug("{} images found ".format(total_matches))
            result = []
            for offset in range(0, total_matches, count):
                images = self._make_api_call(q,offset,count)
                result += images
                time.sleep(timestamp)
            for batch in more_itertools.chunked(result, batch_size):
                delayed_tasks = []
                for img_uri in batch:
                    try:
                        if FileUtil.exists_http_file(img_uri):
                            delayed_tasks.append(dask.delayed(ImageUtil.url2img)(img_uri))
                    except Exception as ex:
                        if type(ex) in EXCEPTIONS:
                            logger.debug("skipping: {}".format(img_uri))
                        else:
                            logger.debug("skipping {}: {}".format(img_uri, ex))
                        continue
                compute_result =  dask.compute(*delayed_tasks)
                yield [img for img in compute_result if isinstance(img, np.ndarray)]
        except Exception as ex:
            logger.error("error fetching the images:  {}".format(ex))
