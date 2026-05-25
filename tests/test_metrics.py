import numpy as np
import pytest
import torch

from lidar_owl.metrics import AnomalyMetric, CalibrationMetric, SemSegMetricExt


def _one_hot_scores(labels, num_classes):
    scores = np.zeros((len(labels), num_classes), dtype=np.float32)
    scores[np.arange(len(labels)), labels] = 1.0
    return torch.as_tensor(scores)


class DummyWriter:
    def __init__(self):
        self.scalars = {}
        self.texts = {}
        self.figures = {}

    def add_scalar(self, tag, scalar_value, step):
        self.scalars[tag] = (float(scalar_value), step)

    def add_text(self, tag, text_string, global_step=None):
        self.texts[tag] = (text_string, global_step)

    def add_figure(self, tag, figure, global_step=None, close=True):
        self.figures[tag] = (figure, global_step, close)


def test_semseg_metric_ext_confusion_matrix_metrics():
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    preds = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)
    metric = SemSegMetricExt(label_names=["a", "b", "c"], class_categories=[[0, 20], [1, 21], [2, 22]])

    metric.update(_one_hot_scores(preds, num_classes=3), labels)

    np.testing.assert_allclose(metric.acc(), [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(metric.recall(), [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(metric.iou(), [1 / 3, 1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(metric.precision(), [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(metric.f1(), [0.5, 0.5, 0.5, 0.5])
    assert metric.support() == [2, 2, 2]
    np.testing.assert_allclose(metric.cer(), [2 / 3, 2 / 3, 2 / 3, 2 / 3])


def test_calibration_metric_perfect_predictions():
    labels = torch.tensor([0] * 20 + [1] * 20, dtype=torch.long)
    scores = _one_hot_scores(labels.numpy(), num_classes=2)
    metric = CalibrationMetric(label_names=["a", "b"], ece_bins=2)

    metric.update(scores, labels)

    np.testing.assert_allclose(metric.ece(), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(metric.brier_score(), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(metric.ause_brier(), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(metric.ause_miou(), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(metric.uiou(), [1.0, 1.0, 1.0])


def test_calibration_metric_ece_and_brier_score():
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.3, 0.7],
        ],
        dtype=torch.float32,
    )
    metric = CalibrationMetric(label_names=["a", "b"], ece_bins=2)

    metric.update(scores, labels)

    np.testing.assert_allclose(metric.ece(), [0.2, 0.15, 0.175], atol=1e-6)
    np.testing.assert_allclose(metric.brier_score(), [0.2625, 0.2625, 0.525], atol=1e-6)


def test_calibration_metric_uncertainty_metrics_are_finite_for_mixed_predictions():
    labels = torch.tensor([0, 1, 1, 0] * 10, dtype=torch.long)
    scores = torch.tensor(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.3, 0.7],
        ]
        * 10,
        dtype=torch.float32,
    )
    metric = CalibrationMetric(label_names=["a", "b"])

    metric.update(scores, labels)

    for values in (metric.ause_brier(), metric.ause_miou(), metric.uiou()):
        assert len(values) == 3
        assert np.isfinite(values[-1])


def test_calibration_metric_reset_clears_state():
    metric = CalibrationMetric(label_names=["a", "b"])
    metric.update(_one_hot_scores([0, 1], num_classes=2), torch.tensor([0, 1]))
    assert metric.ece()

    metric.reset()

    assert metric.ece() == []
    assert metric.brier_score() == []

def test_anomaly_metric_with_explicit_scores():
    metric = AnomalyMetric(anomaly_label_inds=[2], ignored_label_inds=[0])
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    scores = _one_hot_scores([0, 1, 1, 1], num_classes=2)
    anomaly_scores = torch.tensor([0.1, 0.2, 0.9, 0.8], dtype=torch.float32)

    metric.update(scores, labels, anomaly_scores=anomaly_scores)

    assert metric.auroc() == pytest.approx(1.0)
    assert metric.auprc() == pytest.approx(1.0)
    assert metric.fpr95() == pytest.approx(0.0)
    summary = metric.summary()
    assert summary["auroc"] == pytest.approx(1.0)
    assert summary["auprc"] == pytest.approx(1.0)
    assert summary["fpr95"] == pytest.approx(0.0)
    assert summary["num_points"] == 3
    assert summary["num_anomalies"] == 2


def test_anomaly_metric_default_score_uses_predictive_uncertainty():
    metric = AnomalyMetric(anomaly_label_inds=[2])
    labels = torch.tensor([1, 2, 2, 1], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.95, 0.05],
            [0.55, 0.45],
            [0.60, 0.40],
            [0.90, 0.10],
        ],
        dtype=torch.float32,
    )

    metric.update(scores, labels)

    stored_scores, stored_targets = metric._stored
    np.testing.assert_allclose(stored_scores, [0.05, 0.45, 0.40, 0.10], atol=1e-6)
    np.testing.assert_array_equal(stored_targets, [0, 1, 1, 0])
    assert metric.auroc() == pytest.approx(1.0)


def test_anomaly_metric_returns_nan_without_positive_or_negative_examples():
    metric = AnomalyMetric(anomaly_label_inds=[2])
    labels = torch.tensor([1, 1], dtype=torch.long)
    scores = _one_hot_scores([0, 0], num_classes=2)

    metric.update(scores, labels)

    assert np.isnan(metric.auroc())
    assert np.isnan(metric.fpr95())
    assert np.isnan(metric.auprc())


