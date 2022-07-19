import os
import typing
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

import cv2
import numpy as np
import validators
from PIL import Image as PILImage


class ImageUtils:
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
            # image = np.asarray(bytearray(resp.read()), dtype="uint8")
            # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PILImage.open(resp).convert("RGB")
            image = np.asarray(image)
            return image
        except HTTPError as err:
            if err.code == 404:
                raise Exception("Image not found")
            elif err.code in [403, 406]:
                raise Exception("Forbidden image, it can not be reached")
            else:
                raise

    @classmethod
    def read(cls, image_path):
        if isinstance(image_path, str) and validators.url(image_path):
            image_arr = cls.url2img(image_path)
        elif isinstance(image_path, Path) or os.path.isfile(image_path):
            image_path = str(image_path)
            image_arr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, np.ndarray):
            image_arr = image_path
        else:
            raise Exception("Invalid image type")
        return image_arr

    @staticmethod
    def get_concat_h(im1, im2):
        dst = PILImage.new("RGB", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    @staticmethod
    def get_concat_v(im1, im2):
        dst = PILImage.new("RGB", (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    @staticmethod
    def get_concat_h_multi_resize(im_list, resample=PILImage.BICUBIC):
        min_height = min(im.height for im in im_list)
        im_list_resize = [
            im.resize(
                (int(im.width * min_height / im.height), min_height), resample=resample
            )
            for im in im_list
        ]
        total_width = sum(im.width for im in im_list_resize)
        dst = PILImage.new("RGB", (total_width, min_height))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += im.width
        return dst

    @staticmethod
    def get_concat_v_multi_resize(im_list, resample=PILImage.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [
            im.resize(
                (min_width, int(im.height * min_width / im.width)), resample=resample
            )
            for im in im_list
        ]
        total_height = sum(im.height for im in im_list_resize)
        dst = PILImage.new("RGB", (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst

    @classmethod
    def get_concat_tile_resize(cls, im_list_2d, resample=PILImage.BICUBIC):
        im_list_v = [
            cls.get_concat_h_multi_resize(im_list_h, resample=resample)
            for im_list_h in im_list_2d
        ]
        return cls.get_concat_v_multi_resize(im_list_v, resample=resample)

    @classmethod
    def resize_to_size(cls, image: typing.Union[np.ndarray, PILImage.Image], target_size: tuple, with_padding = False, background_color = 0):
        image = cls.numpy_to_pillow(image)
        in_w, in_h = image.size
        out_w, out_h = target_size
        padding = [0,0]
        if in_w > in_h:
            scale_ratio = out_w / in_w
        else:
            scale_ratio = out_h / in_h
        out_image = image.resize((int(in_w * scale_ratio), int(in_h * scale_ratio)))
        # add padding
        in_w, in_h = out_image.size
        if with_padding:
            if in_w > in_h:
                padding[1] = (in_w - in_h) // 2
                result = PILImage.new(out_image.mode, (in_w, in_w), background_color)
                result.paste(out_image, (0, padding[1]))
            else:
                padding[0] = (in_h - in_w) // 2
                result = PILImage.new(out_image.mode, (in_h, in_h), background_color)
                result.paste(out_image, (padding[0], 0))

            if out_image.palette:
                result.putpalette(out_image.palette)
            out_image = result
        return out_image, scale_ratio, padding

    @classmethod
    def numpy_to_pillow(cls, image):
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        return image

