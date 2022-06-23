![FalconCV Logo](assets/logo.png)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.^-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)

[![GitHub forks](https://img.shields.io/github/forks/haruiz/FalconCV.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/haruiz/FalconCV/network/)
[![GitHub stars](https://img.shields.io/github/stars/haruiz/FalconCV.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/haruiz/FalconCV/stargazers/)
[![GitHub watchers](https://img.shields.io/github/watchers/haruiz/FalconCV.svg?style=social&label=Watch&maxAge=2592000)](https://GitHub.com/haruiz/FalconCV/watchers/)

# FalconCV

FalconCV is an open-source Python library that offers developers a high-level API to interact with some of the most popular computer vision frameworks, such as TensorFlow Object detection API and Detectron.

The main objective behind it is to unify the set of tools available and simplify the use of them. This library is focused mainly on Computer Vision practitioners but also is flexible enough to allow researchers to configure the models at a low-level.

# Supported frameworks

- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

# Installation

### Option 1: Install FalconCV from Pip

```bash
pip install falconcv
```
### Option 2: Install FalconCV from GitHub using Poetry

```bash
git clone https://github.com/haruiz/FalconCV
cd FalconCV
poetry install
```
# Usage

## Datasets Module

### OpenImages example

```python
from falconcv.data import OpenImages
from falconcv.util import FileUtils
from pathlib import Path
import uuid

if __name__ == '__main__':
    # Create the dataset
    ds = OpenImages(
        version=6,  # versions 5 and 6 supported
        split="train",
        task="detection"
    )
    print(ds.home())  # print dataset home    
    print(ds.available_labels)
    images_folder = Path("./images")
    images_folder.mkdir(exist_ok=True)
    FileUtils.clear_folder(images_folder)
    # Download images
    for batch_images in ds.fetch(labels=["cat", "Dog"], n_images=100, batch_size=32):
        for image in batch_images:
            filename = f"{str(uuid.uuid4())}.jpg" # generate unique name
            image.save(images_folder.joinpath(filename))  # save image to disk
```

## Models module

### TensorFlow Object Detection example

**Training a custom model:**

```python
from falconcv.models import ModelBuilder

if __name__ == '__main__':
    config = {
        "checkpoint_uri": "<Model uri from the zoo>",
        "pipeline_uri": "<Config uri from zoo>",
        "images_folder": "<Images folder path>",
        "output_folder": "<output dir>",
    }
    with ModelBuilder.build(config=config) as model:
        model.train(epochs=5000, ratio=0.8, batch_size=32)
        # model.to_saved_model()
        # model.to_tflite()
```

**Inference using a trained model**

```python
from falconcv.models import ModelBuilder
if __name__ == '__main__':
    with ModelBuilder.build("<saved model path>", "<labels map file>.pbx") as model:        
        img = model("<image file|uri>", threshold=0.5)
        img.plot(model.labels_map)   
```

# How to contribute

We are encouraging anyone around the world to contribute to this project. So, we principally need help improving the documentation, translation to other languages (which includes but not limited to French, Spanish, Portuguese, Arabian, and more) or adding new features.
Fork the repository and run the steps from [Install FalconCV from GitHub source using poetry](#option-2-install-falconcv-from-gitHub-using-poetry). Any questions, do not hesitate to write an email to henryruiz22@gmail.com. We are excited to see where this project goes.

Send a pull request!

# Contributors
<a href="https://github.com/haruiz/FalconCV/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=haruiz/FalconCV" />
</a>

# Citation

```commandline
@misc {FalconCV,
    author = "Henry Ruiz, David Lopera",
    title  = "FalconCV, an open-source transfer learning library that offers developers a high level API to interact with some of the most popular computer vision frameworks",
    url    = "https://github.com/haruiz/FalconCV",
    month  = "jun",
    year   = "2020--"
}
```
# Credits
- [Speed/accuracy trade-offs for modern convolutional object detectors.](https://arxiv.org/abs/1611.10012)

# License
Free software: [MIT license](LICENSE)
