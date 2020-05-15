class ModelInfo(object):
    def __init__(self):
        self._name = None
        self._url = None
        self._config_url = None
        self._task = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value):
        self._url = value

    @property
    def config_url(self):
        return self._config_url

    @config_url.setter
    def config_url(self, value):
        self._config_url = value

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    def __str__(self):
        return f"  model: {self.name} - task: {self.task}"

    def __repr__(self):
        return (
            f"ModelInfo(name={self.name}, "
            f"url={self.url}, "
            f"config_url={self.config_url}, "
            f"task={self.task})")
