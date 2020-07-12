from falconcv.cons import *
from falconcv.models import ModelBuilder
from falconcv.models.detectron import ModelZoo


def train_model(model_name, train_images_folder, test_images_folder, out_folder, labels_map,
                epochs=2000, learning_rate=0.002, batch_size=32):
    try:
        config = {
            "model": model_name,
            "train_images_folder": train_images_folder,
            "test_images_folder": test_images_folder,
            "output_folder": out_folder,
            "labels_map": labels_map
        }
        with ModelBuilder.build(config=config, backend=DETECTRON) as model:
            model.train(epochs=epochs, lr=learning_rate, bs=batch_size, clear_folder=True)
    except Exception as ex:
        raise Exception(f"[ERROR] Error training the model {ex}") from ex


if __name__ == '__main__':
    train_images_folder = "/mnt/D/Dev/falconcv/datasets/openimages/animals/train"
    test_images_folder = "/mnt/D/Dev/falconcv/datasets/openimages/animals/test"
    model_folder = "/mnt/D/Dev/falconcv/models/detectron/animals"
    labels_map = {
        "Bird": 1,
        "Eagle": 2,
        "Falcon": 3
    }

    # pick model from zoo
    ModelZoo.print_available_models(task="detection")

    # train model
    model = "R101"
    train_model(model, train_images_folder, test_images_folder, model_folder, labels_map, epochs=500, batch_size=32)
