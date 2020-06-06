import os
from pathlib import Path

from falconcv.models.tf import ModelZoo
from falconcv.ds import *
from falconcv.util import FileUtil, VIUtil, ColorUtil, ImageUtil
from falconcv.models import ModelBuilder
import numpy as np
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

def preprocess(input_image, height, width):
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)
    return image

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

def test_fps_openvino(ir_model,label_map, video_path, device="CPU"):
    from openvino.inference_engine import IENetwork, IECore
    file_name = ir_model.stem
    xml_model_file = ir_model.parent.joinpath("{}.xml".format(file_name))
    bin_model_file = ir_model.parent.joinpath("{}.bin".format(file_name))
    ie = IECore()
    #ver = ie.get_versions(device)[device]
    print("[DEBUG]: Loading model")
    net = ie.read_network(model=str(xml_model_file), weights=str(bin_model_file))
    exec_net = ie.load_network(network=net, device_name=device)
    for k, l in net.inputs.items():
        print(k, net.inputs[k].shape)
    print("[DEBUG]: List Unsupported layers")
    supported_layers = ie.query_network(net, device)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        print(not_supported_layers)
    print("[DEBUG]: Checking output layers")
    output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
    for output_key in net.outputs:
        if net.layers[output_key].type == "DetectionOutput":
            output_name, output_info = output_key, net.outputs[output_key]
    input_blob_img = "image_tensor"
    input_blob_info = "image_info"
    n, c, h, w = net.inputs[input_blob_img].shape
    vs = FileVideoStream(video_path).start()
    time.sleep(2.0)
    fps = FPS().start()
    while vs.more():
        frame = vs.read()
        if not vs.more():
            continue
        height = frame.shape[0]
        width = frame.shape[1]
        frame_tensor = preprocess(frame, h, w)
        res = exec_net.infer({input_blob_img: frame_tensor, input_blob_info: [w, h, 1]})
        res = res[output_name]
        for pred in res[0, 0, :, :]:
            confidence = pred[2]
            box = pred[3:]
            if confidence > 0.5:
                left = int(box[0] * width)
                top = int(box[1] * height)
                right = int(box[2] * width)
                bottom = int(box[3] * height)
                draw_box(frame, None, left, top, right, bottom)
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
    # train model+
    #train_model("faster_rcnn_inception_v2_coco", images_folder, epochs=1000)
    frozen_model = list(Path("./models").rglob("frozen_inference_graph.pb"))[0]
    openVINO_model = list(Path("./models").rglob("frozen_inference_graph.xml"))[0]
    label_map = list(Path("./models").rglob("label_map.pbtxt"))[0]
    test_fps_with_falcon(str(frozen_model), str(label_map), "./../assets/video.mp4")

    #convert_to_openVINO("faster_rcnn_inception_v2_coco", images_folder)
    #test_fps_openvino(openVINO_model, str(label_map), "./../assets/video.mp4")




