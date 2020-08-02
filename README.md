![FalconCV Logo](assets/logo.png)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![Build Docker Image](https://github.com/haruiz/FalconCV/workflows/Build%20Docker%20Image/badge.svg)
![TensorFlow Requirement: 1.15](https://img.shields.io/badge/TensorFlow%20Requirement-1.15-brightgreen)

[![GitHub forks](https://img.shields.io/github/forks/haruiz/FalconCV.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/haruiz/FalconCV/network/)
[![GitHub stars](https://img.shields.io/github/stars/haruiz/FalconCV.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/haruiz/FalconCV/stargazers/)
[![GitHub watchers](https://img.shields.io/github/watchers/haruiz/FalconCV.svg?style=social&label=Watch&maxAge=2592000)](https://GitHub.com/haruiz/FalconCV/watchers/)

# FalconCV

FalconCV is an open-source Python library that offers developers an interface to interact with some of the most popular computer vision frameworks, such as TensorFlow Object detection API and Detectron.

The main objective behind it is to unify the set of tools available and simplify the use of them. This library is focused mainly on Computer Vision practitioners but also is flexible enough to allow researchers to configure the models at a low-level.

Additionally, taking advantage of the fantastic features that OpenVINO offers, a custom model can be trained and optimized to run efficiently in the target hardware with just a few lines of code. It is important to say that FalconCV does not attempt to replace any of the tools mentioned previously; instead, it takes the best of them and offers a solution to improve accessibility to new users.

# Supported frameworks

- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

# Installation

1 - Create and activate a conda environment:

```bash
conda create --name falconcv python=3.6
```
2 - Install dependencies:

```bash
conda install -c conda-forge opencv
conda install -c conda-forge requests
conda install -c conda-forge clint
conda install -c conda-forge git
conda install -c conda-forge gitpython
conda install -c conda-forge validators
conda install -c conda-forge colorama
conda install -c conda-forge tqdm
conda install -c conda-forge boto3
conda install -c conda-forge pillow
conda install -c conda-forge dask
conda install -c conda-forge matplotlib
conda install -c conda-forge lxml
conda install -c conda-forge colorlog
conda install -c conda-forge bs4
conda install -c conda-forge more-itertools
conda install -c conda-forge mako
conda install -c conda-forge wxpython
conda install scikit-learn
conda install -c conda-forge pycairo
```

**Linux:**

```bash
sudo apt install protobuf-compiler
conda install -c conda-forge pycocotools
```

**Windows:**

```bash
conda install -c conda-forge pycocotools
```

3 - Install backends:

- **TensorFlow:** 
```bash
conda install tensorflow-gpu==1.15.0
conda install -c conda-forge tf-slim
```


### Option 1: Install FalconCV from GitHub source

```bash
git clone https://github.com/haruiz/FalconCV
cd FalconCV
python setup.py develop --user
```

### Option 2: Install FalconCV from PyPi (test):

```bash
pip install -i https://test.pypi.org/simple/ falconcv
```

### Option 3: Install FalconCV from github:

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

**Inference using a trained model**

```python
from falconcv.models import ModelBuilder
from falconcv.util import VIUtil
if __name__ == '__main__':
    with ModelBuilder.build("<Frozen model path>.pb", "<labels map file>.pbx") as model:
        image_file = "<image file>.jpg"
        img, predictions = model(image_file, threshold=0.5)
        fig = VIUtil.imshow(img, predictions)
        fig.savefig('demo.png', bbox_inches='tight')
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
