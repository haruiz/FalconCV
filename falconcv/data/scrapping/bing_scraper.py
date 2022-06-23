import logging
import os
import time

import requests
from matplotlib import pyplot as plt

from falconcv.data.scrapping.scraper import ImagesScraper
from falconcv.decor import exception
from falconcv.util import VisUtils

logger = logging.getLogger("rich")
BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"


class BingScraper(ImagesScraper):
    def __init__(self, subscription_key=None, **kwargs):
        super(BingScraper, self).__init__()
        self.subscription_key = (
            subscription_key
            if subscription_key
            else os.environ.get("BING_SEARCH_SUBSCRIPTION_KEY")
        )
        self._request_parameters = kwargs.get(
            "request_parameters",
            {"imageType": "Photo", "license": "All", "size": "Large"},
        )

    def _get_total_matches(self, q):
        resp = requests.get(
            BING_SEARCH_ENDPOINT,
            params={"q": q, **self._request_parameters},
            headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
        )
        response = resp.json()
        if resp.status_code == 200:
            return response["totalEstimatedMatches"]
        elif resp.status_code in [401, 410, 429]:
            raise Exception(response["error"]["message"])
        else:
            raise Exception("\n".join([str(e) for e in response["errors"]]))

    def _make_request(self, q, offset, count):
        images = []
        resp = requests.get(
            BING_SEARCH_ENDPOINT,
            params={
                **self._request_parameters,
                "q": q,
                "count": count,
                "offset": offset,
            },
            headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
        )
        if resp.status_code == 200:
            resp = resp.json()
            for img in resp["value"]:
                img_uri = img["contentUrl"]
                images.append(img_uri)
        elif resp.status_code in [401, 410, 429]:
            raise Exception(resp.json()["error"]["message"])
        return images

    @exception
    def fetch(self, q, n=200, count=50, batch_size: int = 50, timestamp=1):
        total_matches = self._get_total_matches(q)
        logger.info(f"{total_matches} images found for query {q}")
        n = min(n, total_matches)
        images_found = self._get_list_of_images(count, n, q, timestamp)

        for i in range(0, len(images_found), batch_size):
            images_batch = images_found[i : i + batch_size]
            yield self._download_images(images_batch)

    def _get_list_of_images(self, count, n, q, timestamp):
        images_found = []
        for offset in range(0, n, count):
            images_found += self._make_request(q, offset, count)
            time.sleep(timestamp)
        return images_found


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    bing_client = BingScraper()
    with VisUtils.with_matplotlib_backend("tkagg"):
        for images_batch in bing_client.fetch("pomeranian", batch_size=80):
            for img in images_batch:
                plt.imshow(img)
                plt.show()
