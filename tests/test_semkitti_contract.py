import numpy as np
import pytest


open3d = pytest.importorskip("open3d")
assert open3d is not None

from src.lidar_owl.log import project, semkitti_cmap, semkitti_train_id_to_name


def test_semkitti_name_contract_known_ids():
    names = semkitti_train_id_to_name(20)
    assert len(names) == 20
    assert names[0] == "unlabeled"
    assert names[19] == "traffic-sign"


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
