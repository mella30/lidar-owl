import warnings

import numpy as np
from open3d.ml.torch.modules import metrics as ml3d_metrics

import lidar_owl.util as util


class SemSegMetricExt(ml3d_metrics.SemSegMetric):
    """Extended semantic-segmentation metrics compatible with Open3D-ML."""

    def __init__(self, label_names):
        super().__init__()

        # TODO: check this at init
        if label_names is None:
            raise ValueError("label_names must be provided from the dataset.")
        if len(label_names) == 0:
            raise ValueError("label_names must not be empty.")
        self.label_names = list(label_names)

    def update(self, scores, labels):
        super().update(scores, labels)
        # add metrics here which cannot be derived from the confusion matrix 

    def acc(self):
        acc = super().acc()
        return [] if acc is None else acc

    def iou(self):
        iou = super().iou()
        return [] if iou is None else iou

    def precision(self):
        if self.confusion_matrix is None:
            return []

        precision = util.safe_divide(
            np.diag(self.confusion_matrix),
            self.confusion_matrix.sum(axis=0),
        )
        macro_precision = float(np.nanmean(precision)) if precision.size else float("nan")
        return precision.tolist() + [macro_precision]

    def recall(self):
        return self.acc()

    def f1(self):
        if self.confusion_matrix is None:
            return []

        precision = np.asarray(self.precision()[:-1], dtype=np.float64)
        recall = np.asarray(self.recall()[:-1], dtype=np.float64)
        f1 = util.safe_divide(2 * precision * recall, precision + recall)
        macro_f1 = float(np.nanmean(f1)) if f1.size else float("nan")
        return f1.tolist() + [macro_f1]

    def support(self):
        if self.confusion_matrix is None:
            return []
        return self.confusion_matrix.sum(axis=1).astype(np.int64).tolist()

    def summary(self):

        acc = self.acc()
        recall = self.recall()
        iou = self.iou()
        precision = self.precision()
        f1 = self.f1()
        support = self.support()

        per_class = []
        for idx in range(self.num_classes):
            per_class.append(
                {
                    "index": idx,
                    "name": self.label_names[idx],
                    "accuracy": acc[idx],
                    "iou": iou[idx],
                    "precision": precision[idx],
                    "recall": recall[idx],
                    "f1": f1[idx],
                    "support": support[idx],
                }
            )

        return {
            "confusion_matrix": self.confusion_matrix.copy(),
            "per_class": per_class,
            "overall_accuracy": acc[-1],
            "mean_iou": iou[-1],
            "macro_precision": precision[-1],
            "macro_recall": recall[-1],
            "macro_f1": f1[-1],
        }

