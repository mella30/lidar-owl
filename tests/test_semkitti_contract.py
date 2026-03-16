import numpy as np
import pytest
import torch
import yaml
from pathlib import Path


open3d = pytest.importorskip("open3d")
assert open3d is not None

from src.lidar_owl.log import (
    compact_label_names_from_dataset,
    project,
    restore_prediction_labels,
    semkitti_cmap,
    semkitti_train_id_to_name,
)
from src.lidar_owl.losses import CrossEntropyFlat


def _semkitti_resource():
    return Path(open3d._ml3d.__file__).parent / "datasets" / "_resources" / "semantic-kitti.yaml"


def _semkitti_data():
    with _semkitti_resource().open() as f:
        return yaml.safe_load(f)


def test_semkitti_name_contract_known_ids():
    names = semkitti_train_id_to_name(20)
    assert len(names) == 20
    assert names[0] == "unlabeled"
    assert names[19] == "traffic-sign"


def test_compact_semkitti_metric_names_skip_ignored_unlabeled():
    class DummyDataset:
        label_to_names = {idx: name for idx, name in enumerate(semkitti_train_id_to_name(20))}

    names = compact_label_names_from_dataset(DummyDataset(), 19, ignored_label_inds=[0])

    assert len(names) == 19
    assert names[0] == "car"
    assert names[-1] == "traffic-sign"


def test_semkitti_palette_shape_and_range():
    palette = semkitti_cmap(20)
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

    img = project(points, labels, palette, size=(64, 64), axes=(0, 1), depth_axis=2)
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
        project(points, labels, palette, size=(32, 32), axes=(0, 1), depth_axis=2)


def test_project_keeps_same_frame_when_ignore_mask_differs():
    points = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    gt = np.array([[1], [1], [1], [1]], dtype=np.int64)
    pred = np.array([[1], [1], [1], [0]], dtype=np.int64)
    palette = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    gt_img = project(points, gt, palette, size=(21, 5), axes=(0, 1), depth_axis=2)
    pred_img = project(points, pred, palette, size=(21, 5), axes=(0, 1), depth_axis=2)

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
        size=(21, 5),
        axes=(0, 1),
        depth_axis=2,
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
    palette = semkitti_cmap(20)

    gt_img = project(points, gt, palette, size=(64, 64), axes=(0, 1), depth_axis=2)
    pred_img = project(
        points,
        restore_prediction_labels(pred_compact, ignored_label_inds=[0]),
        palette,
        size=(64, 64),
        axes=(0, 1),
        depth_axis=2,
    )

    assert gt_img is not None
    assert pred_img is not None
    np.testing.assert_allclose(pred_img, gt_img)


def test_semkitti_label_handling_pipeline():
    data = _semkitti_data()
    learning_map = {int(k): int(v) for k, v in data["learning_map"].items()}
    learning_map_inv = {int(k): int(v) for k, v in data["learning_map_inv"].items()}
    learning_ignore = {int(k): bool(v) for k, v in data["learning_ignore"].items()}

    raw_target = torch.tensor([0, 1, 10, 13, 252, 60, 81, 99], dtype=torch.long)
    expected_train_target = torch.tensor([0, 0, 1, 5, 1, 9, 19, 0], dtype=torch.long)
    train_target = torch.tensor([learning_map[int(label)] for label in raw_target], dtype=torch.long)

    assert torch.equal(train_target, expected_train_target)
    assert all(learning_ignore[int(label)] == (label == 0) for label in train_target.tolist())

    logits = torch.full((len(raw_target), 20), -6.0, dtype=torch.float32)
    for row, label in enumerate(train_target.tolist()):
        logits[row, label] = 6.0
    logits.requires_grad_(True)

    loss_mod = CrossEntropyFlat(ignore_index=0)
    loss = loss_mod(logits, train_target)
    assert torch.isfinite(loss)

    mutated_logits = logits.detach().clone()
    mutated_logits[0] = 0.0
    mutated_logits[1] = torch.linspace(-50.0, 50.0, steps=20)
    mutated_loss = loss_mod(mutated_logits, train_target)
    assert torch.isclose(loss.detach(), mutated_loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad[0], torch.zeros_like(logits.grad[0]))
    assert torch.allclose(logits.grad[1], torch.zeros_like(logits.grad[1]))
    assert torch.count_nonzero(logits.grad[2:]).item() > 0

    pred_train = logits.detach().argmax(dim=1)
    pred_raw = torch.tensor([learning_map_inv[int(label)] for label in pred_train], dtype=torch.long)
    expected_pred_raw = torch.tensor([0, 0, 10, 20, 10, 40, 81, 0], dtype=torch.long)

    assert torch.equal(pred_train, train_target)
    assert torch.equal(pred_raw, expected_pred_raw)
