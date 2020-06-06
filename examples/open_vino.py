import os
from pathlib import Path

from falconcv.models.tf import ModelZoo
from falconcv.ds import *
from falconcv.util import FileUtil, VIUtil, ColorUtil, ImageUtil
from falconcv.models import ModelBuilder

#video test
import time
import cv2
import imutils
from falconcv.models import ModelBuilder
from imutils.video import VideoStream, FPS, FileVideoStream


def create_dataset(images_folder, labels, n, batch_size):
    try:
        # creating dataset
        dataset = OpenImages(v=6)
        dataset.setup(split="train", task="detection")
        # labels = dataset.labels_map.values() # get valid labels
        os.makedirs(images_folder, exist_ok=True)
        FileUtil.clear_folder(images_folder)
        for batch_images in dataset.fetch(
                n=n,
                labels=labels,
                batch_size=batch_size):
            for img in batch_images:
                img.export(images_folder)
                for region in img.regions:
                    pass
                    # print(region.shape_attributes["x"],region.shape_attributes["y"])
    except Exception as ex:
        print("error creating the dataset {} ".format(ex))


def train_model(model_name, images_folder, epochs=5000):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder
        }
        with ModelBuilder.build(config=config) as model:
            model.train(epochs=epochs, val_split=0.3, clear_folder=True)
            model.freeze(epochs)
    except Exception as ex:
        raise Exception("Error training the model {} ".format(ex)) from ex


def convert_to_openVINO(model_name, images_folder):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder
        }
        with ModelBuilder.build(config=config) as model:
            model.to_OpenVINO(device="CPU")
    except Exception as ex:
        raise Exception("Error training the model {} ".format(ex)) from ex


def draw_box(img, label, x, y, x_plus_w, y_plus_h):
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (23, 230, 210), 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (23, 230, 210), 2)


def test_fps_with_falcon(frozen_model,label_map, video_path):
    vs = FileVideoStream(video_path).start()
    time.sleep(2.0)
    fps = FPS().start()
    with ModelBuilder.build(frozen_model, label_map) as model:
        while vs.more():
            frame = vs.read()
            if not vs.more():
                continue
            frame, boxes = model.predict(frame, threshold=0.5, top_k=10)
            for box in boxes:
                draw_box(frame, None, box.x1, box.y1, box.x2, box.y2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # update the fps counter
            fps.update()
        # stop the timer and display the information
        fps.stop()
        print("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        vs.stop()

if __name__ == '__main__':
    images_folder = "/home/haruiz/datasets/eagles"
    model_name = "faster_rcnn_inception_v2_coco"
    # create the dataset
    #create_dataset(images_folder, labels=["Eagle"], n=1000, batch_size=250)
    # train model
    #train_model("faster_rcnn_inception_v2_coco", images_folder, epochs=1000)
    frozen_model = list(Path("./models").rglob("frozen_inference_graph.pb"))[0]
    label_map = list(Path("./models").rglob("label_map.pbtxt"))[0]
    #test_fps_with_falcon(str(frozen_model), str(label_map), "./../assets/video.mp4")
    convert_to_openVINO("faster_rcnn_inception_v2_coco", images_folder)



