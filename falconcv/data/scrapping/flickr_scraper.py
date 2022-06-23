import logging
import logging
import math
import os
import time

import matplotlib.pyplot as plt
import requests

from falconcv.data.scrapping.scraper import ImagesScraper
from falconcv.decor import exception
from falconcv.util import VisUtils

FLICKR_SEARCH_API_ENDPOINT = "https://api.flickr.com/services/rest"
logger = logging.getLogger("rich")


class FlickrScraper(ImagesScraper):
    def __init__(self, api_key=None, **kwargs):
        super(FlickrScraper, self).__init__()
        self._api_key = api_key if api_key else os.environ.get("FLICKR_SEARCH_API_KEY")
        self._request_parameters = kwargs.get(
            "request_parameters",
            {
                "api_key": self._api_key,
                "method": "flickr.photos.search",
                "tag_mode": "any",
                # "privacy_filter": "1"
                "content_type": 1,
                "media": "photos",
                "per_page": 0,
                "nojsoncallback": 1,
                "format": "json",
            },
        )

    def _get_total_matches(self, q):
        response = requests.get(
            url=FLICKR_SEARCH_API_ENDPOINT,
            params={"tags": q, **self._request_parameters},
        )
        if response.status_code == 200:
            json_object = response.json()
            if json_object["stat"] == "ok":
                total_matches = int(json_object["photos"]["total"])
                return total_matches
            else:
                raise Exception(json_object["message"])

    def _make_request(self, q, count, page, sz):
        resp = requests.get(
            url=FLICKR_SEARCH_API_ENDPOINT,
            params={
                **self._request_parameters,
                "tags": q,
                "per_page": count,
                "page": page,
                "extras": ",".join(
                    ["url_o", "url_k", "url_h", "url_l", "url_c", "url_m"]
                ),
            },
        )
        images = []
        if resp.status_code == 200:
            resp = resp.json()
            if resp["stat"] == "ok":
                for photo in resp["photos"]["photo"]:
                    if sz in photo:
                        images.append(photo[sz])
        return images

    def _get_list_of_images(self, count, n, q, timestamp, sz):
        images_found = []
        number_of_pages = math.ceil(n / count)
        for page in range(1, number_of_pages + 1):
            images_found += self._make_request(q, count, page, sz)
            time.sleep(timestamp)
        return images_found

    @exception
    def fetch(self, q, n=200, count=50, batch_size: int = 100, timestamp=1, sz="url_m"):
        assert batch_size <= 500, "invalid count parameter"
        total_matches = self._get_total_matches(q)
        logger.info(f"{total_matches} images found for query {q}")
        n = min(n, total_matches)
        images_found = self._get_list_of_images(count, n, q, timestamp, sz)

        for i in range(0, len(images_found), batch_size):
            images_batch = images_found[i : i + batch_size]
            yield self._download_images(images_batch)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    flickr_client = FlickrScraper()
    with VisUtils.with_matplotlib_backend("tkagg"):
        for images_batch in flickr_client.fetch("pomeranian", batch_size=80):
            for img in images_batch:
                plt.imshow(img)
                plt.show()
