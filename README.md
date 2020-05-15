![FalconCV Logo](assets/logo.png)
--------------------------------------

# FalconCV

FalconCV is an open-source Python library that offers developers an interface to interact with some of the most popular computer vision frameworks, such as TensorFlow Object detection API and Detectron.

The main objective behind it is to unify the set of tools available and simplify the use of them. This library is focused mainly on Computer Vision practitioners but also is flexible enough to allow researchers to configure the models at a low-level.

Additionally, taking advantage of the fantastic features that OpenVINO offers, a custom model can be trained and optimized to run efficiently in the target hardware with just a few lines of code. It is important to say that FalconCV does not attempt to replace any of the tools mentioned previously; instead, it takes the best of them and offers a solution to improve accessibility to new users.

# Supported frameworks

- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

# Installation

## Option 1. Install FalconCV from GitHub source:

1 - Create and activate a conda environment:

```bash
conda create --name falconcv python=3.6
```

2 - Clone FalconCV repository using `git`:

```bash
git clone https://github.com/haruiz/FalconCV
cd FalconCV folder
```

3 - Install dependencies:

```bash
pip install matplotlib
pip install numpy==1.17
pip install opencv-contrib-python
pip install pillow
pip install cython
pip install tqdm
pip install scipy
pip install requests
pip install clint
pip install validators
pip install more-itertools
pip install pandas
pip install imutils
pip install boto3
pip install "dask[complete]"
pip install lxml
pip install Mako
pip install colorlog
pip install colorama
pip install bs4
pip install pick
pip install -U scikit-learn
pip install gitpython
pip install omegaconf
```

**Linux:**

```bash
conda install -c anaconda wxpython
sudo apt install protobuf-compiler
pip install pycocotools
```

**Windows:**

```bash
pip install windows-curses
pip install pycocotools-win
```

4 - Install backends:

- **TensorFlow:** `conda install tensorflow-gpu==1.15.0`

## Option 2. Install FalconCV from PyPi (test):

```bash
pip install -i https://test.pypi.org/simple/ falconcv
```

# Usage

## Datasets

### OpenImages example

```python
from falconcv.ds import *

if __name__ == '__main__':
    # Create the dataset
    dataset = OpenImages(v=6) # versions 5 and 6 supported
    dataset.setup(split="train", task="detection")
    images_folder = "<output folder>"
    for batch_images in dataset.fetch(
        n=100, # number of images by class
        labels=["Mouse", "dog"], # target labels
        batch_size=50 # batch images size
        ):
        # Do something cool with the images
        for img in batch_images:
            # export images to disk
            img.export(images_folder)
            for region in img.regions:
                print(region.shape_attributes["x"],
                      region.shape_attributes["y"])
```

## Models

### TensorFlow Object Detection example

**Train a custom model:**

```python
from falconcv.models import ModelBuilder
from falconcv.util import VIUtil
import falconcv as fcv

if __name__ == '__main__':
    config = {
        "model": model_name,
        "images_folder": images_folder,
        "output_folder": out_folder,
        "labels_map": labels_map,
    }
    with ModelBuilder.build(config=config) as model:
        model.train(epochs=epochs,
                    val_split=0.3,
                    clear_folder=False)
        # for freezing the model
        model.freeze(chekpoint=epochs)
```

For more detailed info visit the [documentation](https://haruiz.github.io/FalconCV/).

Also, you can open the Colab demo:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q_l7RsAFiITJVj8yOMLR0yVNf97T7r43)

# API Documentation

You can found the API documentation at <https://haruiz.github.io/FalconCV/>.

# Roadmap

[Detailed Roadmap](https://github.com/github/hub/projects/1)

# How to contribute

Fork the repository and then run the steps from [Install FalconCV from GitHub source](#option-1-install-falconcv-from-github-source).

Send a pull request!

# License

Free software: [MIT license](LICENSE)

Citation: haruiz. FalconCV. Git Code (2020). <https://github.com/haruiz/FalconCV/>
