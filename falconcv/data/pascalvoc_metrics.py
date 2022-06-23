from collections import namedtuple

import pandas as pd
from alive_progress import alive_bar
import numpy as np

DetectionMatch = namedtuple("DetectionMatch", "gt_idx pred_idx iou")


class PascalVOCMetricsHandler:
    def __init__(self, model, eval_dataset, labels, confidence_threshold=0.5):
        self._model = model
        self._eval_dataset = eval_dataset
        self._confidence_threshold = confidence_threshold
        self._labels = labels

    @classmethod
    def __detections_to_matches(
        cls, groundtruth_annotations, predicted_annotations, iou_threshold=0.5
    ):
        matches = []
        # found iou matches
        for i in range(len(groundtruth_annotations)):
            for j in range(len(predicted_annotations)):
                groundtruth_box = groundtruth_annotations[i]
                prediction = predicted_annotations[j]
                iou = groundtruth_box.iou(prediction)
                if iou > iou_threshold:
                    matches.append(DetectionMatch(gt_idx=i, pred_idx=j, iou=iou))
        # annotations that are considered as true positives
        if len(matches) > 0:
            matches = cls.__remove_repeat_matches(matches)
        return matches

    @staticmethod
    def __remove_repeat_matches(matches):
        # remove gt and pred repeat boxes
        matches = list(sorted(matches, key=lambda x: x.iou, reverse=False))
        matches = list({m.gt_idx: m for i, m in enumerate(matches)}.values())
        matches = list(sorted(matches, key=lambda x: x.iou, reverse=False))
        matches = list({m.pred_idx: m for i, m in enumerate(matches)}.values())
        return matches

    def compute_confusion_matrix(self, iou_threshold=0.5):
        cols = self._labels + ["undetected"]
        indices = self._labels + ["misdetected"]
        confusion_matrix = pd.DataFrame(0, columns=cols, index=indices)
        with alive_bar(len(self._eval_dataset), bar="bubbles") as bar:
            for image in self._eval_dataset:
                input_image = image
                output_image = self._model(
                    image.image_path, threshold=self._confidence_threshold
                )
                groundtruth_annotations = input_image.annotations
                predicted_annotations = output_image.annotations
                matches = self.__detections_to_matches(
                    groundtruth_annotations, predicted_annotations, iou_threshold
                )
                if len(matches) > 0:
                    for i in range(len(groundtruth_annotations)):
                        groundtruth_box = groundtruth_annotations[i]
                        filter_matches = list(filter(lambda a: a.gt_idx == i, matches))
                        # count the number of true and false predictions
                        if len(filter_matches) > 0:
                            confusion_matrix.loc[
                                groundtruth_box.name, groundtruth_box.name
                            ] += 1
                        # count the number of undetected objects
                        else:
                            confusion_matrix.loc[
                                groundtruth_box.name, "undetected"
                            ] += 1

                    for i in range(len(predicted_annotations)):
                        prediction_box = predicted_annotations[i]
                        filter_matches = list(
                            filter(lambda a: a.pred_idx == i, matches)
                        )
                        # count the number of misdetections
                        if len(filter_matches) == 0:
                            confusion_matrix.loc[
                                "misdetected", prediction_box.name
                            ] += 1
                del matches
                del input_image
                del output_image
                bar()
        return confusion_matrix

    def compute_precision_recall(self, iou_threshold=0.5, confusion_matrix=None):
        precision, recall = {}, {}
        if confusion_matrix is None:
            confusion_matrix = self.compute_confusion_matrix(iou_threshold)
        for cat in self._labels:
            total_annotations = confusion_matrix.loc[cat, :].sum()  # row
            total_predictions = confusion_matrix.loc[:, cat].sum()  # column
            precision_by_cat = float(confusion_matrix.loc[cat, cat] / total_predictions)
            recall_by_call = float(confusion_matrix.loc[cat, cat] / total_annotations)
            precision[cat] = round(precision_by_cat, 2)
            recall[cat] = round(recall_by_call, 2)
        return precision, recall

    def compute_f1_score(self, iou_threshold=0.5, confusion_matrix=None):
        precision, recall = self.compute_precision_recall(
            iou_threshold, confusion_matrix
        )
        f1_score = {}
        for cat in self._labels:
            f1_score[cat] = round(
                2 * (precision[cat] * recall[cat]) / (precision[cat] + recall[cat]), 2
            )
        return f1_score

    def compute_mAP(self, iou_threshold=0.5, confusion_matrix=None):
        precision, recall = self.compute_precision_recall(
            iou_threshold, confusion_matrix
        )
        mAP = 0
        for cat in self._labels:
            mAP += precision[cat]
        mAP = mAP / len(self._labels)
        return round(mAP, 2)

    def compute_mAP_per_ious(
        self,
        iou_thresholds=np.arange(start=0.3, stop=1.0, step=0.05),
        confusion_matrix=None,
    ):
        mAP_per_ious = {}
        for iou_threshold in iou_thresholds:
            mAP_per_ious[iou_threshold] = self.compute_mAP(
                iou_threshold, confusion_matrix
            )
        return mAP_per_ious

    def compute_precision_recall_f1_score_mAP(
        self, iou_threshold=0.5, confusion_matrix=None
    ):
        precision, recall = self.compute_precision_recall(
            iou_threshold, confusion_matrix
        )
        f1_score = self.compute_f1_score(iou_threshold, confusion_matrix)
        mAP = self.compute_mAP(iou_threshold, confusion_matrix)
        return precision, recall, f1_score, mAP
