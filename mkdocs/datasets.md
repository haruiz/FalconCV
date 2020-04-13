# Datasets

**OpenImages** 

Open Images is a dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives. It contains a total of 16M bounding boxes for 600 object classes on 1.9M images, making it the largest existing dataset with object location annotations. The boxes have been largely manually drawn by professional annotators to ensure accuracy and consistency. [^1] 
[^1]:  https://github.com/openimages/dataset

 
!!! note "Open Images"
    ```python
    from falconcv.ds import *
    if __name__ == '__main__':          
        # creating dataset
        dataset = OpenImages(v=6) #5/6 supported
        dataset.setup(split="train", task="detection")
        #labels = dataset.labels_map.values() # get valid labels
        images_folder = "<output folder>"         
        for batch_images in dataset.fetch(
                n=100, # number of images by class
                labels=["Mouse", "dog"],# target labels
                batch_size=50# batch images size
                ):
            # Do something cool with the images
            for img in batch_images:
                # export images to disk 
                img.export(images_folder)
                for region in img.regions:
                    print(region.shape_attributes["x"], region.shape_attributes["y"])
    ```


**Cocos**

COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled) [^2]
[^2]:  http://cocodataset.org/

!!! note "COCO (Common Objects in Context)"
    ```python
    from falconcv.ds import *
    if __name__ == '__main__':          
        # creating dataset
        dataset = Coco(v=2017) #Only 2017 supported
        dataset.setup(split="train", task="detection")
        #labels = dataset.labels_map.values() # get valid labels
        images_folder = "<output folder>"         
        for batch_images in dataset.fetch(
                n=100, # number of images by class
                labels=["Mouse", "dog"],# target labels
                batch_size=50# batch images size
                ):
            # Do something cool with the images
            for img in batch_images:
                # export images to disk 
                img.export(images_folder)
                for region in img.regions:
                    print(region.shape_attributes["x"], region.shape_attributes["y"])
    
    ```

**Scrapper**
- Using the falconCV scrapper you can download images from bing and flicker

**Bing Images Scrapper**

!!! info
    How to generate the subscription key?
    https://azure.microsoft.com/en-us/try/cognitive-services/my-apis/?api=search-api-v7

!!! notes "Download images from bing"
    ````python
    from falconcv.ds import *
    import uuid, os
    import cv2
    if __name__ == '__main__': 
        out_folder="<out folder>"
        scrapper=BingScrapper("<Subscription Key>")
        for images_batch in scrapper.fetch(q="cockroach",batch_size=80):
            for img in images_batch:
                # copy the images to the disk or do whatever you want
                cv2.imwrite(os.path.join(out_folder, "{}.jpg".format(str(uuid.uuid4()))), img)
    ````

**Flickr Images Scrapper**

!!!info
    how to generate the api key??
    https://www.flickr.com/services/api/misc.api_keys.html

!!! note "Download images from Flickr"
    ````python
    from falconcv.ds import *
    import uuid, os
    import cv2
    if __name__ == '__main__':
        out_folder=r"<out folder>"
        scrapper=FlickrScrapper(api_key="<Api Key>")
        for batch in scrapper.fetch(q=["cockroach"],batch_size=80):
            for img in batch:
                # copy the images to the disk or do whatever you want
                cv2.imwrite(os.path.join(out_folder,"{}.jpg".format(str(uuid.uuid4()))),img)
    ````
