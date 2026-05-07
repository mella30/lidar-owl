import numpy as np
import pytest
import yaml
import torch
from pathlib import Path
import open3d._ml3d
from open3d.ml.torch.modules import losses as ml3d_losses

from lidar_owl.log import (
    _compact_label_names_from_dataset,
    project,
    _semkitti_cmap,
    _semkitti_train_id_to_name,
)
from lidar_owl.ml3d_util import restore_prediction_labels
from lidar_owl.metrics import SemSegMetricExt


def _semkitti_resource():
    return Path(open3d._ml3d.__file__).parent / "datasets" / "_resources" / "semantic-kitti.yaml"


def _semkitti_data():
    with _semkitti_resource().open() as f:
        return yaml.safe_load(f)


def test_semkitti_name_contract_known_ids():
    names = _semkitti_train_id_to_name(20)
    assert len(names) == 20
    assert names[0] == "unlabeled"
    assert names[19] == "traffic-sign"


def test_compact_semkitti_metric_names_skip_ignored_unlabeled():
    class DummyDataset:
        label_to_names = {idx: name for idx, name in enumerate(_semkitti_train_id_to_name(20))}

    names = _compact_label_names_from_dataset(DummyDataset(), 19, ignored_label_inds=[0])

    assert len(names) == 19
    assert names[0] == "car"
    assert names[-1] == "traffic-sign"


def test_semkitti_palette_shape_and_range():
    palette = _semkitti_cmap(20)
    assert palette.shape == (20, 3)
    assert np.all(palette >= 0.0)
    assert np.all(palette <= 1.0)


def test_project_ignores_label_zero_and_renders_positive_labels():
    points = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 1.0]], dtype=np.float32
    )
    labels = np.array([[0], [1], [2]], dtype=np.int64)
    palette = np.array(
        [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]], dtype=np.float32
    )

    img = project(points, labels, palette, view="bev", bev_size=(64, 64))
    assert img is not None

    # Label 0 should be ignored by projection mask.
    has_label_0_color = np.any(np.all(np.isclose(img, palette[0], atol=1e-6), axis=-1))
    has_label_1_color = np.any(np.all(np.isclose(img, palette[1], atol=1e-6), axis=-1))
    has_label_2_color = np.any(np.all(np.isclose(img, palette[2], atol=1e-6), axis=-1))

    assert not has_label_0_color
    assert has_label_1_color
    assert has_label_2_color


def test_project_out_of_range_label_raises():
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    labels = np.array([[3]], dtype=np.int64)
    palette = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6]], dtype=np.float32)

    with pytest.raises(IndexError):
        project(points, labels, palette, view="bev", bev_size=(32, 32))


def test_project_keeps_same_frame_when_ignore_mask_differs():
    points = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    gt = np.array([[1], [1], [1], [1]], dtype=np.int64)
    pred = np.array([[1], [1], [1], [0]], dtype=np.int64)
    palette = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    gt_img = project(points, gt, palette, view="bev", bev_size=(21, 5))
    pred_img = project(points, pred, palette, view="bev", bev_size=(21, 5))

    assert gt_img is not None
    assert pred_img is not None
    assert np.allclose(gt_img[0, 20], palette[1])
    assert np.allclose(pred_img[0, 20], 0.0)
    assert np.allclose(pred_img[0, 10], palette[1])


def test_project_with_gt_visibility_mask_hides_extra_prediction_points():
    points = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32
    )
    gt = np.array([[1], [1], [0]], dtype=np.int64)
    pred = np.array([[1], [1], [1]], dtype=np.int64)
    palette = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    visible_mask = (gt > 0).reshape(-1)

    pred_img = project(
        points,
        pred,
        palette,
        view="bev",
        bev_size=(21, 5),
        visible_mask=visible_mask,
    )

    assert pred_img is not None
    assert np.allclose(pred_img[0, 20], 0.0)
    assert np.allclose(pred_img[0, 10], palette[1])


def test_restore_prediction_labels_reinserts_ignored_train_ids():
    pred_compact = np.array([[0], [8], [18]], dtype=np.int64)

    restored = restore_prediction_labels(pred_compact, ignored_label_inds=[0])

    np.testing.assert_array_equal(restored, np.array([[1], [9], [19]], dtype=np.int64))


def test_restored_prediction_labels_match_gt_projection_colors():
    points = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [0.0, 10.0, 2.0]], dtype=np.float32
    )
    gt = np.array([[1], [9], [19]], dtype=np.int64)
    pred_compact = np.array([[0], [8], [18]], dtype=np.int64)
    palette = _semkitti_cmap(20)

    gt_img = project(points, gt, palette, view="bev", bev_size=(64, 64))
    pred_img = project(
        points,
        restore_prediction_labels(pred_compact, ignored_label_inds=[0]),
        palette,
        view="bev",
        bev_size=(64, 64),
    )

    assert gt_img is not None
    assert pred_img is not None
    np.testing.assert_allclose(pred_img, gt_img)


def _one_hot_scores(labels, num_classes):
    scores = np.zeros((len(labels), num_classes), dtype=np.float32)
    scores[np.arange(len(labels)), labels] = 1.0
    return torch.as_tensor(scores)


def test_val_metric_contract_perfect_predictions_have_full_miou():
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    metric = SemSegMetricExt(label_names=["class_0", "class_1", "class_2"])

    metric.update(_one_hot_scores(labels, num_classes=3), torch.as_tensor(labels))

    np.testing.assert_allclose(metric.iou(), [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(metric.acc(), [1.0, 1.0, 1.0, 1.0])


def test_val_metric_contract_swapped_predictions_have_zero_miou():
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    predictions = np.array([1, 2, 0, 1, 2, 0], dtype=np.int64)
    metric = SemSegMetricExt(label_names=["class_0", "class_1", "class_2"])

    metric.update(_one_hot_scores(predictions, num_classes=3), torch.as_tensor(labels))

    np.testing.assert_allclose(metric.iou(), [0.0, 0.0, 0.0, 0.0])


def test_val_metric_contract_absent_classes_do_not_change_mean_iou():
    # This protects against an easy-to-miss mIoU artifact: absent classes should be
    # ignored via nanmean, not counted as zero and not counted as perfect.
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    metric = SemSegMetricExt(label_names=["class_0", "class_1", "absent_class"])

    metric.update(_one_hot_scores(labels, num_classes=3), torch.as_tensor(labels))

    iou = metric.iou()
    np.testing.assert_allclose(iou[:2], [1.0, 1.0])
    assert np.isnan(iou[2])
    assert iou[-1] == pytest.approx(1.0)


def test_val_metric_contract_semkitti_ignored_unlabeled_filtering():
    # SemanticKITTI dataset train IDs include unlabeled=0, but the model predicts
    # compact classes 0..18 after Open3D filters ignored_label_inds=[0].
    gt_dataset_train_ids = torch.tensor([0, 1, 9, 19, 0], dtype=torch.long)
    pred_compact_ids = np.array([5, 0, 8, 18, 7], dtype=np.int64)
    raw_scores = _one_hot_scores(pred_compact_ids, num_classes=19)

    valid_scores, valid_labels = ml3d_losses.filter_valid_label(
        raw_scores,
        gt_dataset_train_ids,
        num_classes=19,
        ignored_label_inds=[0],
        device=raw_scores.device,
    )

    metric = SemSegMetricExt(label_names=[f"class_{idx}" for idx in range(19)])
    metric.update(valid_scores, valid_labels)

    iou = np.asarray(metric.iou(), dtype=np.float64)
    np.testing.assert_array_equal(valid_labels.numpy(), np.array([0, 8, 18]))
    np.testing.assert_allclose(iou[[0, 8, 18, -1]], [1.0, 1.0, 1.0, 1.0])
