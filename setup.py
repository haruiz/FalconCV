import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="falconcv",
    version="1.0.22",
    author="Henry Ruiz",
    author_email="henry.ruiz@tamu.edu",
    description="Computer Vision Transfer Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haruiz/FalconCV",
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib',
        'numpy==1.17',
        'opencv-contrib-python',
        'pillow',
        'cython',
        'tqdm',
        'scipy',
        'requests',
        'clint',
        'validators',
        'more-itertools',
        'pandas',
        'imutils',
        'boto3',
        'dask[complete]',
        'lxml',
        'Mako',
        'colorlog',
        'colorama',
        'bs4',
        'pick',
        'scikit-learn',
        'gitpython'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

