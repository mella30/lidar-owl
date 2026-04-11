from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from open3d.ml.torch.modules import losses as ml3d_losses

from lidar_owl.losses import CrossEntropyFlat
from lidar_owl.models import RandLANetFlat, BaseFlatAdapter

open3d = pytest.importorskip("open3d")

def test_semseglossadapter_get_loss_returns_filtered_labels_and_scores():
    model = SimpleNamespace(
        cfg=SimpleNamespace(num_classes=19, ignored_label_inds=[0]),
        custom_loss=CrossEntropyFlat(ignore_index=-1),
    )
    learned_labels = torch.tensor([0, 0, 1, 5, 1, 9, 19, 0], dtype=torch.long)
    results = torch.randn(len(learned_labels), 19, dtype=torch.float32)
    inputs = {"data": {"labels": learned_labels}}

    loss, compact_labels, valid_scores = BaseFlatAdapter.get_loss(
        model,
        Loss=None,
        results=results,
        inputs=inputs,
        device=results.device,
    )

    expected_scores, expected_labels = ml3d_losses.filter_valid_label(
        results,
        learned_labels,
        num_classes=19,
        ignored_label_inds=[0],
        device=results.device,
    )
    expected_loss = F.cross_entropy(expected_scores, expected_labels, ignore_index=-1)

    assert torch.isclose(loss, expected_loss)
    assert torch.equal(compact_labels, expected_labels)
    assert torch.equal(valid_scores, expected_scores)
    assert compact_labels.ndim == 1
    assert valid_scores.shape == (5, 19)


def test_semseglossadapter_falls_back_to_open3d_loss_when_no_custom_loss():
    class DummyOpen3DLoss:
        @staticmethod
        def weighted_CrossEntropyLoss(scores, labels):
            return F.cross_entropy(scores, labels, ignore_index=-1)

    model = SimpleNamespace(
        cfg=SimpleNamespace(num_classes=19, ignored_label_inds=[0]),
        custom_loss=None,
    )
    learned_labels = torch.tensor([0, 0, 1, 5, 1, 9, 19, 0], dtype=torch.long)
    results = torch.randn(len(learned_labels), 19, dtype=torch.float32)
    inputs = {"data": {"labels": learned_labels}}

    loss, compact_labels, valid_scores = BaseFlatAdapter.get_loss(
        model,
        Loss=DummyOpen3DLoss(),
        results=results,
        inputs=inputs,
        device=results.device,
    )

    expected_loss = DummyOpen3DLoss.weighted_CrossEntropyLoss(valid_scores, compact_labels)

    assert torch.isclose(loss, expected_loss)


def test_new_semseg_models_should_put_adapter_before_open3d_model_in_mro():
    # Contract reminder: if you add another Open3D-ML semseg model wrapper,
    # put SemSegLossAdapter before the Open3D model class in the inheritance list.
    assert RandLANetFlat.__mro__[1] is BaseFlatAdapter
