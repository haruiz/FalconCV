try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow not found. Please install TensorFlow.")
from .toda_zoo import TODAZoo
