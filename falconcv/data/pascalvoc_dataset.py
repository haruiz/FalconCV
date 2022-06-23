import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from falconcv.decor import depends
from .pascalvoc_image import PascalVOCImage

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    ...


class PascalVOCDataset:
    def __init__(self, image_files):
        self.image_files = image_files

    @property
    def files(self):
        return self.image_files

    @property
    def labels(self):
        return list(
            set([annotation.name for image in self for annotation in image.annotations])
        )

    @property
    def labels_map(self):
        return {label.lower(): i + 1 for i, label in enumerate(self.labels)}

    def take(self, n):
        return self.__class__(self.image_files[:n])

    def tail(self, n):
        return self.__class__(self.image_files[-n:])

    def head(self, n):
        return self.take(n)

    def empty(self):
        return len(self.image_files) == 0

    def __getitem__(self, index):
        return self.__mk_image(self.image_files[index])

    @classmethod
    def from_folder(cls, folder: typing.Union[Path, str]):
        folder = Path(folder)
        image_files = [
            Path(folder) / f
            for f in folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg"]
        ]
        assert len(image_files) > 0, "No images found in folder {}".format(folder)
        return cls(image_files)

    @classmethod
    def from_list(cls, image_files):
        return cls(image_files)

    @depends("tensorflow")
    def mk_labels_map_file(self, output_file):
        output_file = Path(output_file)
        if output_file.exists():
            output_file.unlink()
        with open(output_file, "a") as f:
            for name, idx in self.labels_map.items():
                item = "item{{\n id: {} \n name: '{}'\n}} \n".format(idx, name.lower())
                f.write(item)

    @depends("tensorflow")
    def mk_record_file(self, output_file):
        labels_map = self.labels_map
        output_file = Path(output_file)
        if output_file.exists():
            output_file.unlink()
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(
                lambda image: image.to_example_train(labels_map), self
            )
            with tf.io.TFRecordWriter(str(output_file)) as writer:
                for example in results:
                    writer.write(example.SerializeToString())

    def batch(self, batch_size):
        for i in range(0, len(self.image_files), batch_size):
            yield self.__class__(self.image_files[i : i + batch_size])

    def sample(self, n):
        return self.__class__(np.random.choice(self.image_files, n, replace=False))

    @staticmethod
    def __mk_image(image_file: Path):
        xml_path = image_file.absolute().with_suffix(".xml")
        mask_path = image_file.absolute().with_suffix(".png")
        if mask_path.exists():
            return PascalVOCImage(image_file, mask_path, xml_path)
        elif xml_path.exists():
            return PascalVOCImage(image_file, xml_path=xml_path)
        else:
            return PascalVOCImage(image_file)

    def __iter__(self):
        for image_file in self.image_files:
            yield self.__mk_image(image_file)

    def __len__(self):
        return len(self.image_files)

    def split(self, ratio, shuffle=True, random_state=None):
        if shuffle:
            if random_state is None:
                random_state = np.random.randint(0, high=2**32 - 2, dtype=np.int64)
            np.random.seed(random_state)
            np.random.shuffle(self.image_files)
        n = int(len(self.image_files) * ratio)
        return self.head(n), self.tail(len(self.image_files) - n)
