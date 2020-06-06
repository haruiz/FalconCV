# Datasets

## Open Images Dataset

Open Images is a dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives. It contains a total of 16M bounding boxes for 600 object classes on 1.9M images, making it the largest existing dataset with object location annotations. The boxes have been largely manually drawn by professional annotators to ensure accuracy and consistency.

FalconCV only supports versions 5 and 6 of Open Images Dataset [^1]
[^1]: https://storage.googleapis.com/openimages/web/index.html


**Setup Method:**

| parameter  | description          | values                  | example                               |
| ---------- | -------------------- | ----------------------- | ------------------------------------- |
| split      | split for dataset    | train, test, validation | split="train" (default: train)        |
| task       | computer vision task | detection               | task="detection" (default: detection) |

**Fetch Method:**

| parameter  | description               | values  | example                       |
| ---------- | ------------------------  | ------- | ----------------------------- |
| n          | number of images by class | int     | n=100                         |
| labels     | target labels             | int     | labels=["Bear", "Elephant"]   |
| batch_size | images to load in memory  | int     | batch_size=100 (default: 200) |

!!! note "Open Images Dataset"
    ```python
    import os
    from falconcv.ds import *
    from falconcv.util import FileUtil, ImageUtil

    if __name__ == '__main__':          
        # create dataset
        dataset = OpenImages(v=6) # versions 5 and 6 supported
        dataset.setup(split="train", task="detection")
        # create ouput folder
        out_folder = "<output folder>"   
        os.makedirs(out_folder, exist_ok=True)
        # optional: clear folder if already exists 
        FileUtil.clear_folder(out_folder)
        for batch_images in dataset.fetch(
                n=100, # number of images by class
                labels=["Bear", "Elephant"], # target labels
                batch_size=100 # n images to load in memory
            ):
            # access to batch images
            for img in batch_images:
                # export images to disk 
                img.export(images_folder)
                for region in img.regions:
                    print(region.shape_attributes["x"], region.shape_attributes["y"])
    ```

## COCO Dataset

COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled).

FalconCV only supports version 2017 of COCO Dataset.[^2]
[^2]: http://cocodataset.org/

**Setup Method:**

| parameter  | description          | values                  | example                               |
| ---------- | -------------------- | ----------------------- | ------------------------------------- |
| split      | split for dataset    | train, validation       | split="train" (default: train)        |
| task       | computer vision task | detection, segmentation | task="detection" (default: detection) |

**Fetch Method:**

| parameter  | description               | values  | example                       |
| ---------- | ------------------------  | ------- | ----------------------------- |
| n          | number of images by class | integer | n=100                         |
| labels     | target labels             | integer | labels=["Bear", "Elephant"]   |
| batch_size | images to load in memory  | integer | batch_size=100 (default: 200) |

!!! note "COCO (Common Objects in Context)"
    ```python
    import os
    from falconcv.ds import *
    from falconcv.util import FileUtil, ImageUtil

    if __name__ == '__main__':          
        # create dataset
        dataset = Coco(v=2017) # only 2017 version is supported
        dataset.setup(split="train", task="detection")
        # create ouput folder
        out_folder = "<output folder>"
        os.makedirs(out_folder, exist_ok=True)
        # optional: clear folder if already exists 
        FileUtil.clear_folder(out_folder)     
        for batch_images in dataset.fetch(
                n=100, # number of images by class
                labels=["Mouse"], # target labels
                batch_size=100 # n images to load in memory
            ):
            # Do something cool with the images
            for img in batch_images:
                # export images to disk 
                img.export(images_folder)
                for region in img.regions:
                    print(region.shape_attributes["x"], region.shape_attributes["y"])
    
    ```

## Scrappers

Use FalconCV's scrappers to download images from Bing and Flicker.

### Bing Images Scrapper

**Fetch Method:**

| parameter  | description                   | values  | example                       |
| ---------- | ----------------------------- | ------- | ----------------------------- |
| q          | images to search              | string  | q="bear"                      |
| count      | number of images              | integer | count=150 (default: 100)      |
| batch_size | images to load in memory      | integer | batch_size=100 (default: 200) |
| timestamp  | time to wait between requests | integer | timestamp=5 (default: 1)      |

!!! info
    How to generate the subscription key?
    https://azure.microsoft.com/en-us/try/cognitive-services/my-apis/?api=search-api-v7

!!! notes "Download images from bing"
    ````python
    import uuid, os
    import cv2
    from falconcv.ds import *

    if __name__ == '__main__': 
        out_folder="<out folder>"
        scrapper=BingScrapper("<Subscription Key>")
        for images_batch in scrapper.fetch(q="cockroach", batch_size=80):
            for img in images_batch:
                # copy the images to the disk or do whatever you want
                cv2.imwrite(os.path.join(out_folder, "{}.jpg".format(str(uuid.uuid4()))), img)
    ````

### Flickr Images Scrapper

**Fetch Method:**

| parameter  | description                     | values  | example                       |
| ---------- | ------------------------------- | ------- | ----------------------------- |
| q          | images to search                | string  | q=["cockroach"]               |
| batch_size | images to load in memory        | integer | batch_size=100 (default: 100) |
| timestamp  | time to wait between requests   | integer | timestamp=5 (default: 1)      |
| sz         | images sizes (check list below) | string  | sz="url_o" (default: url_m)   |

List of sizes:

- url_o: Original (4520 × 3229)
- url_k: Large 2048 (2048 × 1463)
- url_h: Large 1600 (1600 × 1143)
- url_l: Large 1024 (1024 × 732)
- url_c: Medium 800 (800 × 572)
- url_z: Medium 640 (640 × 457)
- url_m: Medium 500 (500 × 357)
- url_n: Small 320 (320 × 229)
- url_s: Small 240 (240 × 171)
- url_t: Thumbnail (100 × 71)
- url_q: Square 150 (150 × 150)
- url_sq: Square 75 (75 × 75)

!!!info
    How to generate the API key??
    https://www.flickr.com/services/api/misc.api_keys.html

!!! note "Download images from Flickr"
    ````python
    import uuid, os
    import cv2
    from falconcv.ds import *

    if __name__ == '__main__':
        out_folder=r"<out folder>"
        scrapper=FlickrScrapper(api_key="<Api Key>")
        for batch in scrapper.fetch(q=["cockroach"], batch_size=80):
            for img in batch:
                # copy the images to the disk or do whatever you want
                cv2.imwrite(os.path.join(out_folder, "{}.jpg".format(str(uuid.uuid4()))), img)
    ````
