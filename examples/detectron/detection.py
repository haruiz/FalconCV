import sys
sys.path.append('.')

from falconcv.models.detectron import DetectronModelZoo


if __name__ == '__main__':
    # pick model from zoo
    DetectronModelZoo.print_available_models(task="detection")
