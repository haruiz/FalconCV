import os
import validators
import numpy as np
import urllib.request
import cv2
from urllib.error import HTTPError


class ImageUtil:
    @staticmethod
    def process_input_image(input_image, size=None):
        img_arr, scale_factor = ImageUtil.read(input_image), 1  # read image
        if size:
            img_arr, scale_factor = ImageUtil.resize(img_arr, width=size[0], height=[1])  # resize image
        img_height, img_width = img_arr.shape[:2]

        return img_arr, img_width, img_height, scale_factor
    
    @staticmethod
    def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        (h,w)=image.shape[:2]
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r=height/float(h)
            dim=(int(w*r),height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r=width/float(w)
            dim=(width,int(h*r))
        # resize the image
        resized=cv2.resize(image,dim,interpolation=inter)
        # return the resized image
        return resized, r

    @staticmethod
    def url2img(url):
        try:
            assert validators.url(url),"invalid url"
            resp=urllib.request.urlopen(url,timeout=30)
            image=np.asarray(bytearray(resp.read()),dtype="uint8")
            image=cv2.imdecode(image,cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except HTTPError as err:
            if err.code == 404:
                raise Exception("Image not found")
            elif err.code == 403:
                raise Exception("Forbidden image, try with other one")
            else:
                raise

    @classmethod
    def read(cls, image):
        image_arr = None
        if isinstance(image,str):
            if os.path.isfile(image):
                image_arr = cv2.imread(image,cv2.IMREAD_COLOR)
            elif validators.url(image):
                image_arr = cls.url2img(image)
            else:
                raise IOError("Invalid image")
            image_arr=cv2.cvtColor(image_arr,cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            image_arr = image
        return image_arr
