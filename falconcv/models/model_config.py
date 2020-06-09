class ModelConfig(object):
    def __init__(self):
        self._model = None
        self._is_pretrained_model = False
        self._is_frozen_model = False
        self._images_folder = None
        self._labels_map = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def is_pretrained_model(self):
        return self._is_pretrained_model

    @is_pretrained_model.setter
    def is_pretrained_model(self, value):
        self._is_pretrained_model = value

    @property
    def is_frozen_model(self):
        return self._is_frozen_model

    @is_frozen_model.setter
    def is_frozen_model(self, value):
        self._is_frozen_model = value

    @property
    def images_folder(self):
        return self._images_folder

    @images_folder.setter
    def images_folder(self, value):
        self._images_folder = value

    @property
    def labels_map(self):
        return self._labels_map

    @labels_map.setter
    def labels_map(self, value):
        self._labels_map = value
