class DatasetUtils:
    @staticmethod
    def get_openimages_deps(version, split):
        deps_map = {
            6: {
                "train": {
                    "class_names": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
                    "images": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
                    "boxes": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
                    "segmentation_classes": "https://storage.googleapis.com/openimages/v5/classes-segmentation.txt",
                    "segmentation": "https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv",
                },
                "test": {
                    "class_names": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
                    "images": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
                    "boxes": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
                    "segmentation": "https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv",
                    "segmentation_classes": "https://storage.googleapis.com/openimages/v5/classes-segmentation.txt",
                },
                "validation": {
                    "class_names": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
                    "images": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
                    "boxes": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
                    "segmentation_classes": "https://storage.googleapis.com/openimages/v5/classes-segmentation.txt",
                    "segmentation": "https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv",
                },
            }
        }
        return deps_map[version][split]

    @staticmethod
    def get_coco_files(version, split):
        deps_map = {
            2017: {
                "train": {
                    "annotations_file_uri": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                },
                "test": {
                    "annotations_file_uri": "http://images.cocodataset.org/annotations/image_info_test2017.zip"
                },
                "validation": {
                    "annotations_file_uri": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                },
            }
        }
        return deps_map[version][split]
