import io
import math
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from lxml import objectify
# from .color_util import ColorUtil
from matplotlib import patches

from .color_util import ColorUtil
from .misc_util import BoundingBox


class VIUtil:

    @staticmethod
    def fig2buffer(fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        return buf

    @classmethod
    def fig2img(cls, fig):
        buf = cls.fig2buffer(fig)
        w, h, d = buf.shape
        img = Image.frombytes("RGBA", (w, h), buf.tostring())
        return np.asarray(img)

    @staticmethod
    def fig2numpy(fig, dpi=180):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _imshow(img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("src", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("src", (800, 600))
        cv2.imshow("src", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def draw_mask(image, mask, color=(255, 0, 0), alpha=0.4):
        assert image.dtype == np.uint8, '`image` not of type np.uint8'
        assert mask.dtype == np.uint8, '`mask` not of type np.uint8'
        if np.any(np.logical_and(mask != 1, mask != 0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        if image.shape[:2] != mask.shape:
            raise ValueError('The image has spatial dimensions %s but the mask has '
                             'dimensions %s' % (image.shape[:2], mask.shape))
        pil_image = Image.fromarray(image)
        solid_color = np.expand_dims(
            np.ones_like(mask), axis=2) * np.reshape(list(color), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        np.copyto(image, np.array(pil_image.convert('RGB')))

    @classmethod
    def imshow(cls, img, boxes, alpha=0.3):
        labels = np.unique([box.label for box in boxes])
        colors_dict = ColorUtil.rainbow(len(labels))
        R = np.asarray(colors_dict["r"], dtype=np.float) / 255.0
        G = np.asarray(colors_dict["g"], dtype=np.float) / 255.0
        B = np.asarray(colors_dict["b"], dtype=np.float) / 255.0
        A = [1.0] * len(labels)
        palette = list(zip(B.tolist(), G.tolist(), R.tolist(), A))
        colors = {label: palette[i] for i, label in enumerate(labels)}
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, aspect='equal')
        for i, box in enumerate(boxes):
            c = colors[box.label]
            if isinstance(box.mask, np.ndarray):
                mask = box.mask
                rgb_color = tuple(map(lambda x: math.floor(x * 255), c[:-1]))
                cls.draw_mask(img, mask, color=rgb_color, alpha=alpha)
            label = "{} :{:.2f}".format(box.label, box.score)
            ax.add_patch(
                patches.Rectangle(
                    (box.x1, box.y1),
                    box.x2 - box.x1,
                    box.y2 - box.y1,
                    linewidth=3,
                    edgecolor=c,
                    facecolor='none',
                    fill=False
                ))
            ax.text(
                x=box.x1,
                y=box.y1,
                s=label,
                color="white",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor=colors[box.label], alpha=0.9))
            ax.set_axis_off()
        ax.imshow(img)
        # img = cls.fig2numpy(fig)
        # cls._imshow(img)
        plt.show()
        # cv2.putText(rgb, label,(box.x1 + 10,box.y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[box.label],2)
        # fig.savefig('result_{}.png'.format(uuid.uuid4()), dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def get_voc_annotations(image_path: str, xml_path: str = None, mask_path=None, labels_map=None):
        '''
        return the annotations of the image
        :param image_path: image path
        :return: a list with the bounding boxes
        '''
        try:
            image_path = Path(image_path)
            file_name = image_path.stem
            if xml_path is None:
                xml_path = str(image_path.parent.joinpath("{}.xml".format(file_name)))

            mask_numpy = None
            try:
                if mask_path:
                    with open(mask_path, 'rb') as fid:
                        encoded_mask_png = fid.read()
                    encoded_png_io = io.BytesIO(encoded_mask_png)
                    pil_mask = Image.open(encoded_png_io)
                    if pil_mask.format != 'PNG':
                        raise ValueError('Mask image format not PNG')
                    mask_numpy = np.asarray(pil_mask)
            except Exception as ex:
                raise Exception("error reading the image")

            # load image
            with open(xml_path) as f:
                xml = f.read()
            root = objectify.fromstring(xml)
            boxes = []
            for item in root.object:
                xmin = item.bndbox.xmin
                ymin = item.bndbox.ymin
                xmax = item.bndbox.xmax
                ymax = item.bndbox.ymax
                label = str(item.name)
                box = BoundingBox(
                    x1=xmin,
                    x2=xmax,
                    y1=ymin,
                    y2=ymax,
                    label=label.title()
                )
                boxes.append(box)
                if isinstance(mask_numpy, np.ndarray) and labels_map:
                    if label in labels_map:
                        class_id = labels_map[label]
                        binary_mask = np.zeros_like(mask_numpy)
                        binary_mask[mask_numpy == class_id] = 1
                        box.mask = binary_mask
                boxes.append(box)
            return boxes
        except Exception as e:
            print("Error reading the image {}".format(str(e)))

    @classmethod
    def make_grid(cls, images_folder, annotations_folder=None, masks_folder=None, n=4, rows=2, figsize=(10, 10),
                  fontsize=10, labels_map=None):
        pass

        # images_folder = Path(images_folder)
        # assert images_folder.exists(), "images folder not found"
        # if annotations_folder is None:
        #     annotations_folder = images_folder
        # if masks_folder is None:
        #     masks_folder = images_folder
        # # read files
        # img_files = FileUtil.get_files(images_folder, [".jpg", ".jpeg"])
        # xml_files = FileUtil.get_files(annotations_folder, [".xml"])
        # png_files = FileUtil.get_files(masks_folder, [".png"])
        # files = img_files + xml_files + png_files
        # files = sorted(files, key=lambda img: img.stem)
        # files = [(img_name,list(img_files)) for img_name, img_files in itertools.groupby(files, key=lambda img: img.stem)]
        # files = random.sample(files, k=n)
        # if labels_map:
        #     labels_map = {k.title():v for k,v in labels_map.items()}
        # # load annotations
        # annotations_dict = {}
        # for img_name, img_files in files:
        #     if len(img_files) >= 2:
        #         img_path = img_files[0]
        #         xml_path = img_files[1]
        #         mask_path = None
        #         if len(img_files) == 3:
        #             mask_path = img_files[2]
        #         annotations_dict[img_path] = cls.get_voc_annotations(img_path, xml_path, mask_path, labels_map)
        #
        # assert len(annotations_dict) > 0, "Not annotations found"
        # labels = set([box.label for boxes in annotations_dict.values() for box in boxes ])
        # labels_colors = dict(zip(labels, ColorUtil.rainbow_rgb(len(labels))))
        #
        # # show annotations
        # cols = math.ceil(n / rows)
        # fig = plt.figure(figsize=figsize)
        # plt.style.use("seaborn")
        # fig.subplots_adjust(hspace=0.3, wspace=0.2)
        # for i, (img_path, img_boxes) in enumerate(annotations_dict.items()):
        #     pil_image: Image = Image.open(img_path)
        #     image_numpy = np.asarray(pil_image).copy()
        #     # create figure
        #     ax = fig.add_subplot(rows, cols, i + 1)
        #     ax.set_axis_off()
        #     for box in img_boxes:
        #         label = box.label
        #         color = labels_colors[label]
        #         ax.add_patch(
        #             patches.Rectangle(
        #                 (box.x1, box.y1),
        #                 box.x2 - box.x1,
        #                 box.y2 - box.y1,
        #                 linewidth=3,
        #                 edgecolor=np.asarray(color) /255,
        #                 facecolor='none',
        #                 fill=False
        #             ))
        #         ax.axis('equal')
        #         if isinstance(box.mask, np.ndarray) and labels_map:
        #             cls.draw_mask(image_numpy, box.mask,color)
        #         # ax.text(
        #         #     x=box[0][0] + 20,
        #         #     y=box[0][1] + 20,
        #         #     s=label,
        #         #     color="white",
        #         #     fontsize=12,
        #         #     bbox=dict(boxstyle="round", facecolor=np.array(color) / 255, alpha=0.9))
        #
        #     ax.imshow(image_numpy)
        # #plt.tight_layout()
        # plt.show()
        # return fig
