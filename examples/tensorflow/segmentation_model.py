import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from falconcv.models.tf import ModelZoo
from falconcv.data import Coco
from falconcv.util import FileUtil, VIUtil, ColorUtil
from falconcv.models import ModelBuilder


def create_dataset(images_folder , labels_map,color_palette, n):
    try:
        # creating dataset
        dataset = Coco(v=2017)
        dataset.setup(split="train", task="segmentation")
        #labels = dataset.labels_map.values() # get valid labels
        os.makedirs(images_folder, exist_ok=True)
        FileUtil.clear_folder(images_folder)
        for batch_images in dataset.fetch(
                n=n,
                labels=list(labels_map.keys()),
                batch_size=500):
            for img in batch_images:
                img.export(images_folder, labels_map, color_palette)
                for region in img.regions:
                    pass
                    # print(region.shape_attributes["x"],
                    #       region.shape_attributes["y"])
    except Exception as ex:
        print("error creating the dataset {} ".format(ex))

def train_model(model_name, images_folder, out_folder, labels_map, epochs=5000):
    try:
        config = {
            "model": model_name,
            "images_folder": images_folder,
            "output_folder": out_folder,
            "labels_map": labels_map,
        }
        with ModelBuilder.build(config=config) as model:
            model.train(epochs=epochs, val_split=0.3, clear_folder=True)
            model.freeze(epochs)
    except Exception as ex:
        raise Exception("Error training the model {} ".format(ex)) from ex


def make_predictions(frozen_model,labels_map_file, image):
    # load freeze model
    with ModelBuilder.build(frozen_model,labels_map_file) as model:
        img,predictions=model.predict(image, threshold=0.5)
        import matplotlib
        matplotlib.use('WXAgg')
        VIUtil.imshow(img,predictions)


if __name__ == '__main__':
    images_folder = "/home/haruiz/datasets/animals_mask"
    model_folder = "/home/haruiz/models/animals_mask"
    labels_map = {
        "bird": 1,
        "cat": 2,
        "zebra": 3,
        "horse": 4
    }
    color_palette = ColorUtil.color_palette(n=len(labels_map))

    # creating the dataset
    create_dataset(images_folder, labels_map,color_palette, n=1000)
    # picking and training the model
    print(ModelZoo.available_models(arch="mask")) # check the models available
    train_model("mask_rcnn_inception_v2_coco",images_folder, model_folder, labels_map)

    #doing inference
    frozen_model_file = os.path.join(model_folder, "export/frozen_inference_graph.pb")
    labels_map_file = os.path.join(model_folder, "label_map.pbtxt")
    make_predictions(frozen_model_file, labels_map_file, "../../examples/images/catsvrsbird2.jpeg")
