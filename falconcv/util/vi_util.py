import itertools
import math
import os
import random
from io import BytesIO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
#from .color_util import ColorUtil
from matplotlib import patches
from .file_util import FileUtil
from .color_util import ColorUtil
from lxml import objectify

class VIUtil:

    @staticmethod
    def fig2buffer(fig):
        fig.canvas.draw()
        w,h=fig.canvas.get_width_height()
        buf=np.fromstring(fig.canvas.tostring_argb(),dtype=np.uint8)
        buf.shape=(w,h,4)
        buf=np.roll(buf,3,axis=2)
        return buf

    @classmethod
    def fig2img(cls,fig):
        buf=cls.fig2buffer(fig)
        w,h,d=buf.shape
        img=Image.frombytes("RGBA",(w,h),buf.tostring())
        return np.asarray(img)

    @staticmethod
    def fig2numpy(fig,dpi=180):
        buf=BytesIO()
        fig.savefig(buf,format="png",dpi=dpi)
        buf.seek(0)
        img_arr=np.frombuffer(buf.getvalue(),dtype=np.uint8)
        buf.close()
        img=cv2.imdecode(img_arr,1)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _imshow(img: np.ndarray):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.namedWindow("src",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("src",(800,600))
        cv2.imshow("src",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def draw_mask(image,mask,color=(255,0,0),alpha=0.4):
        assert image.dtype == np.uint8, '`image` not of type np.uint8'
        assert  mask.dtype == np.uint8, '`mask` not of type np.uint8'
        if np.any(np.logical_and(mask != 1,mask != 0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        if image.shape[:2] != mask.shape:
            raise ValueError('The image has spatial dimensions %s but the mask has '
                             'dimensions %s'%(image.shape[:2],mask.shape))
        pil_image=Image.fromarray(image)
        solid_color=np.expand_dims(
            np.ones_like(mask),axis=2)*np.reshape(list(color),[1,1,3])
        pil_solid_color=Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask=Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
        pil_image=Image.composite(pil_solid_color,pil_image,pil_mask)
        np.copyto(image,np.array(pil_image.convert('RGB')))

    @classmethod
    def imshow(cls,img,boxes, alpha=0.3):
        labels=np.unique([box.label for box in boxes])
        colors_dict=ColorUtil.rainbow(len(labels))
        R=np.asarray(colors_dict["r"], dtype=np.float) / 255.0
        G=np.asarray(colors_dict["g"], dtype=np.float) / 255.0
        B=np.asarray(colors_dict["b"], dtype=np.float) / 255.0
        A = [1.0] * len(labels)
        palette=list(zip(B.tolist(),G.tolist(),R.tolist(), A))
        colors={label: palette[i] for i,label in enumerate(labels)}
        #rgb= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        fig=plt.figure(figsize=(20,10))
        ax=fig.add_subplot(111, aspect='equal')
        for i,box in enumerate(boxes):
            c=colors[box.label]
            if isinstance(box.mask, np.ndarray):
                mask = box.mask
                rgb_color=tuple(map(lambda x: math.floor(x*255),c[:-1]))
                cls.draw_mask(img, mask, color=rgb_color, alpha=alpha)
            label="{} :{:.2f}".format(box.label,box.score)
            ax.add_patch(
                patches.Rectangle(
                    (box.x1,box.y1),
                    box.x2-box.x1,
                    box.y2-box.y1,
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
                bbox=dict(boxstyle="round",facecolor=colors[box.label],alpha=0.9))
            ax.set_axis_off()
        ax.imshow(img)
        #img = cls.fig2numpy(fig)
        #cls._imshow(img)
        plt.show()
        #cv2.putText(rgb, label,(box.x1 + 10,box.y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[box.label],2)
        #fig.savefig('result_{}.png'.format(uuid.uuid4()), dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def get_voc_annotations(image_path: str, xml_path: str):
        '''
        return the annotations of the image
        :param image_path: image path
        :return: a list with the bounding boxes
        '''
        try:
            with open(xml_path) as f:
                xml = f.read()
            root = objectify.fromstring(xml)
            boxes = []
            for item in root.object:
                xmin = item.bndbox.xmin
                ymin = item.bndbox.ymin
                xmax = item.bndbox.xmax
                ymax = item.bndbox.ymax
                boxes.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": item.name
                })
            return boxes
        except Exception as e:
            print("Error reading the image {}".format(str(e)))

    @classmethod
    def make_grid(cls, images_folder, annotations_folder=None, n=10, rows=3, figsize = (10,5), fontsize=10):
        import matplotlib
        matplotlib.use('WXAgg')
        import matplotlib.pylab as plt
        images_folder = Path(images_folder)
        assert images_folder.exists(), "images folder not found"
        if annotations_folder is None:
            annotations_folder = images_folder
        img_files = FileUtil.get_files(images_folder, [".jpg", ".jpeg", ".png"])
        xml_files = FileUtil.get_files(annotations_folder, [".xml"])
        files = img_files + xml_files
        files = sorted(files, key=lambda img: img.stem)
        files = [(img_name,list(img_files)) for img_name, img_files in itertools.groupby(files, key=lambda img: img.stem)]
        files = random.sample(files, k=n)
        cols = math.ceil(n/rows)
        fig = plt.figure(figsize=figsize)
        annotations= {img_files[0]:
                          cls.get_voc_annotations(str(img_files[0]), str(img_files[1]))
                                for img_name, img_files in files}
        labels = set([box["label"] for img_path, img_anns in annotations.items() for box in img_anns if box])
        labels_colors = dict(zip(labels, ColorUtil.rainbow_rgb(len(labels))))
        for i, (img_path, img_annotations) in enumerate(annotations.items()):
            ax = fig.add_subplot(rows, cols, i +1)
            ax.set_axis_off()
            img = Image.open(img_path)
            dctx = ImageDraw.Draw(img)  # create drawing context
            for annot in img_annotations:
                box = [(annot["xmin"],annot["ymin"]), (annot["xmax"], annot["ymax"])]
                label = annot["label"]
                color = labels_colors[label]
                dctx.rectangle(box,outline=color, width=5)
                ax.text(
                    x=annot["xmin"] + 20,
                    y=annot["ymin"] + 20,
                    s=label,
                    color="white",
                    fontsize=12,
                    bbox=dict(boxstyle="round",facecolor=np.array(color) / 255,alpha=0.9))
            del dctx
            ax.imshow(np.asarray(img))
        plt.show()



