from contextlib import contextmanager


class VisUtils:
    @staticmethod
    @contextmanager
    def with_matplotlib_backend(backend="tkagg"):
        import matplotlib

        curr_backend = matplotlib.get_backend()
        try:
            matplotlib.use(backend)
            yield
        finally:
            matplotlib.use(curr_backend)
