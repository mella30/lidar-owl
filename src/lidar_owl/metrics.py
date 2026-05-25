from __future__ import annotations

import re

import numpy as np
from open3d.ml.torch.modules import metrics as ml3d_metrics

import lidar_owl.util as util


CLASS_CATEGORIES = [
    [0, 20],
    [1, 20],
    [2, 20],
    [3, 20],
    [4, 21],
    [5, 21],
    [6, 22],
    [7, 22],
    [8, 22],
    [9, 23],
    [10, 23],
    [11, 24],
    [12, 24],
    [13, 24],
    [14, 24],
    [15, 24],
    [16, 25],
    [17, 25],
    [18, 25],
]


def _mean(values: np.ndarray) -> float:
    # Computes a nan-safe mean and returns nan for empty arrays.
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if values.size and not np.all(np.isnan(values)) else float("nan")


def _safe_name(label_names: list[str], idx: int) -> str:
    # Creates TensorBoard-safe class names while keeping the class index visible.
    name = label_names[idx] if idx < len(label_names) else f"class_{idx}"
    return f"{idx:02d}_{re.sub(r'[^0-9A-Za-z_.-]+', '_', name).strip('_')}"


def _display_name(label_names: list[str], idx: int) -> str:
    # Returns a human-readable class label for table rows.
    return label_names[idx] if idx < len(label_names) else f"class_{idx}"


def _format_table_value(value, percent: bool = True) -> str:
    # Formats table values like the original logger: percentages for scores.
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "nan"
    return f"{value * 100.0:.2f}" if percent else f"{value:.0f}"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    # Builds a GitHub-style Markdown table without requiring tabulate as dependency.
    def fmt_cell(value) -> str:
        return str(value).replace("\n", " ")

    header = "| " + " | ".join(fmt_cell(cell) for cell in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(fmt_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _table_rows(label_names: list[str], class_count: int, metrics: dict[str, list], percent_metrics: set[str] | None = None) -> list[list[str]]:
    # Creates per-class rows plus an "all" row for metrics returning class values + overall.
    percent_metrics = set(metrics) if percent_metrics is None else percent_metrics
    rows = []
    for idx in range(class_count):
        rows.append(
            [
                _display_name(label_names, idx),
                *[
                    _format_table_value(values[idx], percent=name in percent_metrics) if len(values) > idx else "n/a"
                    for name, values in metrics.items()
                ],
            ]
        )
    rows.append(
        [
            "all",
            *[
                _format_table_value(values[-1], percent=name in percent_metrics) if values and name in percent_metrics else "n/a"
                for name, values in metrics.items()
            ],
        ]
    )
    return rows


def _log_metric_table(writer, tag: str, label_names: list[str], class_count: int, metrics: dict[str, list], percent_metrics: set[str] | None = None):
    # Logs one class-wise metric table to TensorBoard's Text tab.
    writer.add_text(
        tag,
        _markdown_table(["class", *metrics.keys()], _table_rows(label_names, class_count, metrics, percent_metrics)),
        global_step=0,
    )


def _log_confusion_matrix(writer, tag: str, label_names: list[str], confusion_matrix, step: int = 0):
    # Logs the row-normalized confusion matrix as a TensorBoard image. Rows are
    # ground truth, columns are predictions (Open3D-ML convention).
    if confusion_matrix is None:
        writer.add_text(tag, "No confusion matrix available.", global_step=step)
        return

    figure = _draw_confusion_matrix(confusion_matrix, label_names)
    writer.add_figure(tag, figure, global_step=step, close=True)


def _draw_confusion_matrix(confusion_matrix, label_names: list[str], dtype=np.int32):
    # Draws a compact confusion-matrix figure similar to hmc_semseg's draw_cm:
    # row-normalized percentages, dots for zeros, blue heatmap.
    import matplotlib.pyplot as plt

    cm = np.asarray(confusion_matrix, dtype=np.float32)
    cm = cm / (cm.sum(axis=-1, keepdims=True) + np.finfo(np.float32).tiny)
    cm_scaled = (cm * 100.0).astype(dtype)
    class_count = int(cm_scaled.shape[0])
    xlabels = [_display_name(label_names, idx) for idx in range(cm_scaled.shape[1])]
    ylabels = [_display_name(label_names, idx) for idx in range(class_count)]

    figure = plt.figure(figsize=(4, 4), dpi=320)
    plt.imshow(cm_scaled, cmap=plt.cm.Blues)
    plt.xticks(np.arange(len(xlabels)), xlabels, fontsize=5, rotation=90)
    plt.yticks(np.arange(len(ylabels)), ylabels, fontsize=5)

    for i, j in np.ndindex(cm_scaled.shape):
        value = cm_scaled[i, j]
        if value == 0:
            text = "."
        elif np.issubdtype(cm_scaled.dtype, np.integer):
            text = f"{value}"
        else:
            text = f"{value:.2f}"
        plt.text(j, i, text, ha="center", va="center", color="black", fontsize=3)

    axes = plt.gca()
    axes.set_xlim([-0.5, len(xlabels) - 0.5])
    axes.set_ylim([len(ylabels) - 0.5, -0.5])
    plt.setp(axes.spines.values(), color="black", lw=0.75)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    return figure


def _log_single_row_table(writer, tag: str, row_name: str, metrics: dict[str, object], percent_metrics: set[str] | None = None):
    # Logs one single-row metric table to TensorBoard's Text tab.
    percent_metrics = set(metrics) if percent_metrics is None else percent_metrics
    row = [row_name, *[_format_table_value(value, percent=name in percent_metrics) for name, value in metrics.items()]]
    writer.add_text(tag, _markdown_table(["class", *metrics.keys()], [row]), global_step=0)


def _as_probabilities(scores) -> np.ndarray:
    # Converts logits or probabilities to a 2D probability array.
    scores = util.tensor_to_np(scores)
    scores = np.asarray(scores)
    if scores.size == 0:
        # Empty batches can happen after label filtering.
        return scores.reshape(0, 0)

    scores = scores.reshape(-1, scores.shape[-1]).astype(np.float64, copy=False)
    row_sums = scores.sum(axis=1)
    if np.all(scores >= 0) and np.allclose(row_sums, 1.0, atol=1e-4):
        # Open3D-ML inference usually already returns probabilities.
        return scores

    # Numerically stable softmax for raw logits.
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _class_siblings(num_classes: int, ignored_classes=(), class_categories=CLASS_CATEGORIES) -> list[np.ndarray]:
    # Returns same-parent class groups used by the original non-hierarchical CER.
    ignored = {int(c) for c in ignored_classes}
    parents = {int(child): int(parent) for child, parent in class_categories if int(child) < num_classes}
    siblings = []
    for class_idx in range(num_classes):
        parent = parents.get(class_idx)
        if parent is None:
            siblings.append(np.asarray([class_idx], dtype=np.int64))
            continue
        group = [child for child, group_parent in parents.items() if group_parent == parent and child not in ignored]
        siblings.append(np.asarray(group, dtype=np.int64))
    return siblings


def _calc_brier_sparsification(probs: np.ndarray, labels: np.ndarray, order: np.ndarray, step_percentage: float = 0.01) -> np.ndarray:
    # Mirrors hmc_semseg _calc_brier_sparsification: curve is 1 - remaining Brier error.
    if order.size < 2:
        return np.asarray([], dtype=np.float64)
    probs_sorted = probs[order]
    labels_sorted = labels[order]
    step_size = max(int(np.ceil(order.size * step_percentage)), 1)
    curve = []
    for idx in range(0, order.size - 1, step_size):
        remaining_error = (probs_sorted[idx:-1] - labels_sorted[idx:-1]) ** 2
        curve.append(1.0 - float(np.nanmean(remaining_error)))
    return np.asarray(curve, dtype=np.float64)


def _calc_miou_sparsification(trues: np.ndarray, order: np.ndarray, step_percentage: float = 0.01) -> np.ndarray:
    # Mirrors hmc_semseg _calc_mIoU_sparsification: remaining TP-rate on relevant class points.
    if order.size < 2:
        return np.asarray([], dtype=np.float64)
    correct_sorted = trues[order]
    step_size = max(int(np.floor(order.size * step_percentage)), 1)
    curve = []
    for idx in range(0, order.size - 1, step_size):
        remaining = correct_sorted[idx:-1]
        curve.append(float(np.sum(remaining == 1) / remaining.shape[0]))
    return np.asarray(curve, dtype=np.float64)


class SemSegMetricExt(ml3d_metrics.SemSegMetric):
    """Open3D-ML semantic-segmentation metric plus simple CM-derived scores."""

    def __init__(self, label_names: list[str], ignored_classes=(), class_categories=CLASS_CATEGORIES):
        # Stores label names next to Open3D-ML's confusion-matrix state.
        super().__init__()
        self.label_names = label_names
        self.ignored_classes = {int(c) for c in ignored_classes}
        self.class_categories = class_categories

    @property
    def class_count(self) -> int:
        # Returns the number of evaluated classes, preferring the actual CM size.
        if self.confusion_matrix is not None:
            return int(self.confusion_matrix.shape[0])
        return len(self.label_names)

    def acc(self):
        # Returns per-class recall/accuracy plus Open3D-ML's overall value.
        acc = super().acc()
        return [] if acc is None else acc

    def iou(self):
        # Returns per-class IoU plus Open3D-ML's mean IoU.
        iou = super().iou()
        return [] if iou is None else iou

    def precision(self):
        # Computes per-class and macro precision from the confusion matrix.
        if self.confusion_matrix is None:
            return []
        precision = util.safe_divide(np.diag(self.confusion_matrix), self.confusion_matrix.sum(axis=0))
        return precision.tolist() + [_mean(precision)]

    def recall(self):
        # Uses Open3D-ML's acc() as semantic segmentation recall.
        return self.acc()

    def f1(self):
        # Computes per-class and macro F1 from precision and recall.
        if self.confusion_matrix is None:
            return []
        precision = np.asarray(self.precision()[:-1], dtype=np.float64)
        recall = np.asarray(self.recall()[:-1], dtype=np.float64)
        f1 = util.safe_divide(2 * precision * recall, precision + recall)
        return f1.tolist() + [_mean(f1)]

    def support(self):
        # Counts ground-truth points per class.
        if self.confusion_matrix is None:
            return []
        return self.confusion_matrix.sum(axis=1).astype(np.int64).tolist()

    def cer(self):
        # Computes original non-hierarchical CER: only out-of-category IoU errors are critical.
        if self.confusion_matrix is None:
            return []
        cm = self.confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        union = tp + fp + fn
        siblings = _class_siblings(self.class_count, self.ignored_classes, self.class_categories)
        fn_out = np.zeros(self.class_count, dtype=np.float64)
        fp_out = np.zeros(self.class_count, dtype=np.float64)
        for class_idx in range(self.class_count):
            # Rows are ground truth and columns are predictions in Open3D-ML's CM.
            outside = np.setdiff1d(np.arange(self.class_count), siblings[class_idx], assume_unique=False)
            fn_out[class_idx] = cm[class_idx, outside].sum()
            fp_out[class_idx] = cm[outside, class_idx].sum()
        cer = util.safe_divide(fn_out + fp_out, union)
        return cer.tolist() + [_mean(cer)]

    def summary(self):
        # Packs all performance metrics into a dict for programmatic evaluation.
        values = {
            "accuracy": self.acc(),
            "iou": self.iou(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "support": self.support(),
            "cer": self.cer(),
        }
        per_class = []
        for idx in range(self.class_count):
            per_class.append(
                {
                    "index": idx,
                    "name": self.label_names[idx] if idx < len(self.label_names) else f"class_{idx}",
                    **{name: metric_values[idx] for name, metric_values in values.items() if len(metric_values) > idx},
                }
            )

        # support has no macro element, so it is intentionally omitted from overall.
        overall = {name: metric_values[-1] for name, metric_values in values.items() if metric_values and name != "support"}
        return {
            "confusion_matrix": None if self.confusion_matrix is None else self.confusion_matrix.copy(),
            "per_class": per_class,
            "overall": overall,
            # Backwards-compatible aliases used by earlier experiments.
            "overall_accuracy": overall.get("accuracy", float("nan")),
            "mean_iou": overall.get("iou", float("nan")),
            "macro_precision": overall.get("precision", float("nan")),
            "macro_recall": overall.get("recall", float("nan")),
            "macro_f1": overall.get("f1", float("nan")),
        }

    def log_tensorboard(self, writer, prefix: str):
        # Logs performance metrics as one Markdown table in TensorBoard.
        metrics = {
            "accuracy": self.acc(),
            "mIoU": self.iou(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "cer": self.cer(),
        }
        _log_metric_table(
            writer,
            f"00_performance/{prefix}",
            self.label_names,
            self.class_count,
            metrics,
            percent_metrics=set(metrics),
        )
        _log_confusion_matrix(
            writer,
            f"00_performance/{prefix}_confusion_matrix",
            self.label_names,
            self.confusion_matrix,
            step=0,
        )


class CalibrationMetric:
    """Calibration/uncertainty metrics for filtered semantic predictions."""

    def __init__(self, label_names: list[str], ece_bins: int = 10):
        # Initializes storage for probabilities/labels needed beyond the CM.
        self.label_names = label_names
        self.ece_bins = ece_bins
        self._probs: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._brier_ause_batches: list[float] = []

    @property
    def class_count(self) -> int:
        # Returns the configured number of compact model classes.
        return len(self.label_names)

    @property
    def _stored(self) -> tuple[np.ndarray, np.ndarray]:
        # Concatenates all collected batches into one evaluation array.
        if not self._probs:
            # Keep shapes predictable for downstream metric functions.
            return np.empty((0, self.class_count), dtype=np.float32), np.empty(0, dtype=np.int64)
        return np.concatenate(self._probs, axis=0), np.concatenate(self._labels, axis=0)

    def update(self, scores, labels):
        # Adds one filtered batch of scores and labels to the calibration buffer.
        probs = _as_probabilities(scores)
        labels = util.tensor_to_np(labels).reshape(-1).astype(np.int64)
        if probs.size and labels.size:
            self._probs.append(probs.astype(np.float32, copy=False))
            self._labels.append(labels)
            # Original BrierAUSE is updated per batch and averaged over updates.
            self._brier_ause_batches.append(self._brier_ause_for_arrays(probs, labels))

    def reset(self):
        # Clears all accumulated probabilities and labels.
        self._probs.clear()
        self._labels.clear()
        self._brier_ause_batches.clear()

    def ece(self):
        # Computes expected calibration error overall and by predicted class.
        probs, labels = self._stored
        if probs.size == 0:
            return []
        preds = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        correct = (preds == labels).astype(np.float64)
        per_class = np.array(
            [self._ece_from_confidence(conf[preds == c], correct[preds == c]) for c in range(self.class_count)],
            dtype=np.float64,
        )
        return per_class.tolist() + [self._ece_from_confidence(conf, correct)]

    def _ece_from_confidence(self, conf: np.ndarray, correct: np.ndarray) -> float:
        # Computes scalar ECE from confidences and binary correctness values.
        if conf.size == 0:
            return float("nan")
        edges = np.linspace(0.0, 1.0, self.ece_bins + 1)
        ece = 0.0
        for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            # Include the lower edge only for the first bin to avoid double counting.
            in_bin = (conf >= lo) & (conf <= hi) if idx == 0 else (conf > lo) & (conf <= hi)
            if np.any(in_bin):
                ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
        return float(ece)

    def brier_score(self):
        # Computes multiclass Brier score overall and one-vs-rest per class.
        probs, labels = self._stored
        if probs.size == 0:
            return []
        per_class = []
        for c in range(self.class_count):
            target = (labels == c).astype(np.float64)
            per_class.append(float(np.mean((probs[:, c] - target) ** 2)))
        one_hot = np.eye(self.class_count, dtype=np.float64)[labels]
        return per_class + [float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))]

    def ause_brier(self):
        # Computes original Brier AUSE with confidence/oracle sparsification curves.
        probs, labels = self._stored
        if probs.size == 0:
            return []
        overall = _mean(np.asarray(self._brier_ause_batches, dtype=np.float64))

        per_class = []
        for class_idx in range(self.class_count):
            class_mask = labels == class_idx
            if np.sum(class_mask) < 2:
                per_class.append(float("nan"))
                continue
            per_class.append(self._brier_ause_for_arrays(probs[class_mask], labels[class_mask]))
        return per_class + [overall]

    def ause_miou(self):
        # Computes original mIoU AUSE approximation per class and averages it.
        probs, labels = self._stored
        if probs.size == 0:
            return []
        preds = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        per_class = [self._ause_miou_for_class(preds, labels, confidence, c) for c in range(self.class_count)]
        return per_class + [_mean(np.asarray(per_class, dtype=np.float64))]

    def uiou(self, thresholds: np.ndarray | None = None):
        # Computes uncertainty-aware IoU as best score over confidence thresholds.
        probs, labels = self._stored
        if probs.size == 0:
            return []
        # Match the evaluator convention: sweep confidence thresholds in [0, 1).
        thresholds = np.arange(0.0, 1.0, 0.05) if thresholds is None else thresholds
        preds = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        per_class = []
        for c in range(self.class_count):
            values = []
            for thresh in thresholds:
                certain = conf >= thresh
                tp = np.sum((labels == c) & (preds == c) & certain)
                fp = np.sum((labels != c) & (preds == c) & certain)
                fn = np.sum((labels == c) & (preds != c) & certain)
                true_invalid = np.sum((labels == c) & (preds == c) & ~certain)
                false_invalid = np.sum((labels == c) & (preds != c) & ~certain)
                values.append(util.safe_divide(tp + true_invalid, tp + true_invalid + fp + fn + false_invalid))
            per_class.append(float(np.nanmax(values)) if values else float("nan"))
        return per_class + [_mean(np.asarray(per_class, dtype=np.float64))]

    @staticmethod
    def _brier_ause(probs: np.ndarray, one_hot_labels: np.ndarray, sparse_order: np.ndarray, oracle_order: np.ndarray) -> float:
        # Integrates the gap between Brier sparsification and the original oracle curve.
        sparsification = _calc_brier_sparsification(probs, one_hot_labels, sparse_order)
        oracle = _calc_brier_sparsification(probs, one_hot_labels, oracle_order)
        if sparsification.size == 0 or oracle.size == 0:
            return float("nan")
        min_len = min(sparsification.size, oracle.size)
        return float(np.trapz(np.abs(oracle[:min_len] - sparsification[:min_len]), dx=0.1))

    def _brier_ause_for_arrays(self, probs: np.ndarray, labels: np.ndarray) -> float:
        # Computes the original BrierAUSE update value for one already-masked array.
        if probs.shape[0] < 2:
            return float("nan")
        one_hot = np.eye(self.class_count, dtype=np.float64)[labels]
        preds = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        correctness = (labels == preds).astype(np.float64)
        sparse_order = np.argsort(confidence)
        oracle_order = np.argsort(correctness - confidence**2)
        return self._brier_ause(probs, one_hot, sparse_order, oracle_order)

    def _ause_miou_for_class(self, preds: np.ndarray, labels: np.ndarray, confidence: np.ndarray, class_idx: int) -> float:
        # Computes original class-wise mIoU AUSE on points where pred or label is class_idx.
        mask = (preds == class_idx) | (labels == class_idx)
        if np.sum(mask) <= 19:
            # Match original debug/support guard: skip tiny class slices.
            return float("nan")
        preds_masked = preds[mask]
        labels_masked = labels[mask]
        confidence_masked = confidence[mask]
        # Original treats TPs as true and both FPs/FNs as false on relevant points.
        trues = (preds_masked == class_idx) & (labels_masked == class_idx)
        sparse_order = np.argsort(confidence_masked)
        oracle_order = np.argsort((labels_masked == preds_masked).astype(np.float64))
        sparsification = _calc_miou_sparsification(trues, sparse_order)
        oracle = _calc_miou_sparsification(trues, oracle_order)
        if sparsification.size == 0 or oracle.size == 0:
            return float("nan")
        min_len = min(sparsification.size, oracle.size)
        return float(np.trapz(np.abs(oracle[:min_len] - sparsification[:min_len]), dx=0.1))

    def log_tensorboard(self, writer, prefix: str):
        # Logs calibration and uncertainty metrics as Markdown tables.
        groups = {
            "01_calibration": {
                "ECE": self.ece(),
                "brier_score": self.brier_score(),
                "AUSE_BS": self.ause_brier(),
            },
            "02_uncertainty": {
                "AUSE_mIoU": self.ause_miou(),
                "UIoU": self.uiou(),
            },
        }
        for objective, values_by_metric in groups.items():
            _log_metric_table(writer, f"{objective}/{prefix}", self.label_names, self.class_count, values_by_metric)


class AnomalyMetric:
    """Binary anomaly-detection metrics using uncertainty as anomaly score by default."""

    def __init__(self, anomaly_label_inds=(), ignored_label_inds=()):
        # Stores anomaly and ignore label sets in dataset label space.
        self.anomaly_label_inds = {int(label) for label in anomaly_label_inds}
        self.ignored_label_inds = {int(label) for label in ignored_label_inds} - self.anomaly_label_inds
        self._scores: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, scores, labels, anomaly_scores=None):
        # Adds anomaly scores and binary anomaly targets for one unfiltered batch.
        labels = util.tensor_to_np(labels).reshape(-1).astype(np.int64)
        valid = labels != 255
        if self.ignored_label_inds:
            # Ignore normal ignored labels, but keep labels explicitly configured as anomalies.
            valid &= ~np.isin(labels, list(self.ignored_label_inds))
        if not np.any(valid):
            return

        targets = np.isin(labels[valid], list(self.anomaly_label_inds)).astype(np.int64)
        if anomaly_scores is None:
            # Default novelty score: low semantic confidence means high anomaly score.
            probs = _as_probabilities(scores)
            anomaly_scores = 1.0 - probs.max(axis=1)
        else:
            anomaly_scores = util.tensor_to_np(anomaly_scores).reshape(-1)

        anomaly_scores = anomaly_scores[valid].astype(np.float64, copy=False)
        self._scores.append(anomaly_scores)
        self._targets.append(targets)

    def reset(self):
        # Clears all accumulated anomaly scores and targets.
        self._scores.clear()
        self._targets.clear()

    @property
    def _stored(self) -> tuple[np.ndarray, np.ndarray]:
        # Concatenates stored anomaly batches into flat score/target arrays.
        if not self._scores:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
        return np.concatenate(self._scores), np.concatenate(self._targets)

    def auroc(self) -> float:
        # Computes binary AUROC from ranks of anomaly scores.
        scores, targets = self._stored
        if scores.size == 0 or targets.sum() == 0 or targets.sum() == targets.size:
            # AUROC needs both positive and negative examples.
            return float("nan")
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, scores.size + 1)
        pos_ranks = ranks[targets == 1].sum()
        n_pos = float(np.sum(targets == 1))
        n_neg = float(np.sum(targets == 0))
        return float((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def auprc(self) -> float:
        # Computes area under the precision-recall curve for anomaly detection.
        scores, targets = self._stored
        if scores.size == 0 or targets.sum() == 0:
            # Precision/recall is undefined without positives.
            return float("nan")
        order = np.argsort(-scores)
        y = targets[order]
        tp = np.cumsum(y == 1)
        fp = np.cumsum(y == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / tp[-1]
        precision = np.r_[1.0, precision]
        recall = np.r_[0.0, recall]
        return float(np.trapz(precision, recall))

    def fpr95(self) -> float:
        # Computes false-positive rate at the first threshold reaching about 95% TPR.
        scores, targets = self._stored
        if scores.size == 0 or targets.sum() == 0 or targets.sum() == targets.size:
            # Need both classes to define a false-positive rate.
            return float("nan")
        order = np.argsort(-scores)
        y = targets[order]
        tp = np.cumsum(y == 1)
        fp = np.cumsum(y == 0)
        tpr = tp / np.sum(targets == 1)
        fpr = fp / np.sum(targets == 0)
        # Match original evaluator's rounded criterion (tpr > 0.945).
        idx = np.where(tpr > 0.945)[0]
        return float(fpr[idx[0]]) if idx.size else float("nan")

    def summary(self):
        # Returns all anomaly metrics plus simple support counts.
        scores, targets = self._stored
        return {
            "auroc": self.auroc(),
            "auprc": self.auprc(),
            "fpr95": self.fpr95(),
            "num_points": int(targets.size),
            "num_anomalies": int(targets.sum()),
        }

    def log_tensorboard(self, writer, prefix: str):
        # Logs anomaly metrics as one Markdown table in TensorBoard.
        summary = self.summary()
        _log_single_row_table(
            writer,
            f"03_anomaly/{prefix}",
            "all",
            summary,
            percent_metrics={"auroc", "auprc", "fpr95"},
        )
