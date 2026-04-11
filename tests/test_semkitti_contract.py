import numpy as np
import pytest
import yaml
from pathlib import Path
import open3d._ml3d

from lidar_owl.log import (
    compact_label_names_from_dataset,
    project,
    semkitti_cmap,
    semkitti_train_id_to_name,
)
from lidar_owl.ml3d_util import restore_prediction_labels


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

