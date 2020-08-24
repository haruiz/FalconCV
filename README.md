![FalconCV Logo](assets/logo.png)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![Build Docker Image](https://github.com/haruiz/FalconCV/workflows/Build%20Docker%20Image/badge.svg)
![TensorFlow Requirement: 1.15](https://img.shields.io/badge/TensorFlow%20Requirement-1.15-brightgreen)

[![GitHub forks](https://img.shields.io/github/forks/haruiz/FalconCV.svg?style=social&label=Fork&maxAge=86400)](https://GitHub.com/haruiz/FalconCV/network/)
[![GitHub stars](https://img.shields.io/github/stars/haruiz/FalconCV.svg?style=social&label=Star&maxAge=86400)](https://GitHub.com/haruiz/FalconCV/stargazers/)
[![GitHub watchers](https://img.shields.io/github/watchers/haruiz/FalconCV.svg?style=social&label=Watch&maxAge=86400)](https://GitHub.com/haruiz/FalconCV/watchers/)

# FalconCV

FalconCV is an open-source Python library that offers developers an interface to interact with some of the most popular computer vision frameworks, such as TensorFlow Object detection API and Detectron.

The main objective behind it is to unify the set of tools available and simplify the use of them. This library is focused mainly on Computer Vision practitioners but also is flexible enough to allow researchers to configure the models at a low-level.

Additionally, taking advantage of the fantastic features that OpenVINO offers, a custom model can be trained and optimized to run efficiently in the target hardware with just a few lines of code. It is important to say that FalconCV does not attempt to replace any of the tools mentioned previously; instead, it takes the best of them and offers a solution to improve accessibility to new users.

# Supported frameworks

- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
- [Detectron2](https://github.com/facebookresearch/detectron2).

# Installation

## 1 - Create and activate a conda environment

```bash
conda create --name falconcv python=3.6
```

## 2 - Install dependencies

```bash
conda install -c conda-forge opencv -y
conda install -c conda-forge requests -y
conda install -c conda-forge clint -y
conda install -c conda-forge git -y
conda install -c conda-forge gitpython -y
conda install -c conda-forge validators -y
conda install -c conda-forge colorama -y
conda install -c conda-forge tqdm -y
conda install -c conda-forge boto3 -y
conda install -c conda-forge pillow -y
conda install -c conda-forge dask -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge lxml -y
conda install -c conda-forge colorlog -y
conda install -c conda-forge bs4 -y
conda install -c conda-forge more-itertools -y
conda install -c conda-forge mako -y
conda install -c conda-forge wxpython -y
conda install scikit-learn -y
```

**Linux:**

```bash
sudo apt install protobuf-compiler
conda install -c conda-forge pycocotools
```

**Windows:**

```bash
pip install pycocotools-win
```

## 3 - Install Backends

### **TensorFlow:**

```bash
conda install tensorflow-gpu==1.15.0 -y
conda install -c conda-forge tf-slim -y
```

**Note:** For more details installing TensorFlow go to the [official site](https://www.tensorflow.org/install).

### **Detectron2:**

**PyTorch (PyTorch 1.5.1 + CUDA 10.1):**

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

or

```bash
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**Note:** For more details installing PyTorch go to the [official site](https://pytorch.org/).

**Detectron2 (0.1.3):**

```bash
pip install detectron2==0.1.3 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

**Note:** For more details installing Detectron2 go to the [official site](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

## 4 - Install FalconCV

### **Option 1: Install FalconCV from GitHub source**

```bash
git clone https://github.com/haruiz/FalconCV
cd FalconCV
python setup.py develop --user
```

### **Option 2: Install FalconCV from PyPi (test)**

```bash
pip install -i https://test.pypi.org/simple/ falconcv
```

### **Option 3: Install FalconCV from GitHub**

```bash
pip install git+https://github.com/haruiz/FalconCV.git
```

# Usage

## Datasets

### OpenImages example

```python
from falconcv.data.ds import OpenImages
from falconcv.util import FileUtil
from pathlib import Path

if __name__ == '__main__':
    # Create the dataset
    ds = OpenImages(
        version=6, # versions 5 and 6 supported
        split="train",
        task="detection",
        labels=["cat", "Dog"],# target labels
        n_images=4,# number of images by class
        batch_size=2 # batch images size
    )
    print(ds.home()) # print dataset home
    print(next(ds)) # get next batch
    print(len(ds))
    data_folder = Path("./data")
    data_folder.mkdir(exist_ok=True)
    FileUtil.clear_folder(data_folder)
    # Download images
    for batch_images in ds:
        for image in batch_images:
            image.export(data_folder)  # copy images to disk
```

## Models

### TensorFlow Object Detection example

**Train a custom model:**

```python
from falconcv.models import ModelBuilder

if __name__ == '__main__':
    config = {
        "model": "<model name from zoo>",
        "images_folder": "<images folder path>",
        "output_folder": "<model output folder>"
    }
    with ModelBuilder.build(config=config) as model:
        model.train(epochs=2000,val_split=0.3,clear_folder=False)
```

**Inference using a trained model:**

```python
from falconcv.models import ModelBuilder
from falconcv.util import VIUtil
if __name__ == '__main__':
    with ModelBuilder.build("<Frozen model path>.pb", "<labels map file>.pbx") as model:
        img, predictions = model("<image file|uri>", threshold=0.5)
        VIUtil.imshow(img, predictions)
```

For more detailed info visit the [documentation](https://haruiz.github.io/FalconCV/).
Also, you can open the Colab demo:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q_l7RsAFiITJVj8yOMLR0yVNf97T7r43)

# API Documentation

You can found the API documentation at <https://haruiz.github.io/FalconCV/>.

# Roadmap

You can see the detailed [roadmap](https://github.com/haruiz/FalconCV/projects/1) here.

# How to contribute

We are encouraging anyone around the world to contribute to this project. So, we principally need help improving the documentation, translation to other languages (which includes but not limited to French, Spanish, Portuguese, Arabian, and more) or adding new features.

Fork the repository and run the steps from [Install FalconCV from GitHub source](#option-1-install-falconcv-from-github-source). Any questions, do not hesitate to write an email to henryruiz22@gmail.com. We are excited to see where this project goes.

Send a pull request!

# Contributors

<a href="https://github.com/haruiz/FalconCV/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=haruiz/FalconCV" />
</a>

# Citation

```commandline
@misc {FalconCV,
    author = "Henry Ruiz, David Lopera",
    title  = "FalconCV, an open-source transfer learning library that offers developers an interface to interact with some of the most popular computer vision frameworks",
    url    = "https://github.com/haruiz/FalconCV",
    month  = "jun",
    year   = "2020--"
}
```

# Credits

- [Speed/accuracy trade-offs for modern convolutional object detectors.](https://arxiv.org/abs/1611.10012)

# License

Free software: [MIT license](LICENSE)

# Common Issues

## After installing all the dependencies I still getting the error: `No module named 'tf_slim'

**How to solve it?**

Install `tf_slim` using the command below:

`pip install git+https://github.com/google-research/tf-slim.git`
  