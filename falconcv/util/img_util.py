import os
import urllib.request
from urllib.error import HTTPError

import cv2
import numpy as np
import validators
from PIL import Image


class ImageUtil:
    @staticmethod
    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        (h, w) = image.shape[:2]
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)
        # return the resized image
        return resized, r

    @staticmethod
    def url2img(url):
        try:
            assert validators.url(url), "invalid url"
            resp = urllib.request.urlopen(url, timeout=30)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
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
        if isinstance(image, str):
            if os.path.isfile(image):
                image_arr = cv2.imread(image, cv2.IMREAD_COLOR)
                image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
            elif validators.url(image):
                image_arr = cls.url2img(image)
            else:
                raise IOError("Invalid image")
        elif isinstance(image, np.ndarray):
            image_arr = image
        return image_arr

    @staticmethod
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    @staticmethod
    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    @staticmethod
    def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
        min_height = min(im.height for im in im_list)
        im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height), resample=resample)
                          for im in im_list]
        total_width = sum(im.width for im in im_list_resize)
        dst = Image.new('RGB', (total_width, min_height))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += im.width
        return dst

    @staticmethod
    def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample)
                          for im in im_list]
        total_height = sum(im.height for im in im_list_resize)
        dst = Image.new('RGB', (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst

    @classmethod
    def get_concat_tile_resize(cls, im_list_2d, resample=Image.BICUBIC):
        im_list_v = [cls.get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
        return cls.get_concat_v_multi_resize(im_list_v, resample=resample)
