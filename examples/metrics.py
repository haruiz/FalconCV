from falconcv.models.tf import APIModelFactory
from pathlib import Path

if __name__ == '__main__':
    model_path = Path("/home/haruiz/Filezilla/faster_rcnn_resnet50_coco")
    frozen_model = model_path.joinpath("export/frozen_inference_graph.pb")
    labels_map = model_path.joinpath("label_map.pbtxt")
    eval_record_file = model_path.joinpath("val.record")
    with APIModelFactory.create(str(frozen_model), str(labels_map)) as model:
        confusion_matrix = model.evaluate(
            eval_record_file,
            iou_threshold=0.5,
            confidence_threshold=0.5
        )
        print(confusion_matrix)
