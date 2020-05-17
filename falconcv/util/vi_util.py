import math
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#from .color_util import ColorUtil
from matplotlib import patches

from .color_util import ColorUtil


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
    def img_show(title: str, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, (800, 600))
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
