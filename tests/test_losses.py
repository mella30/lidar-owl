import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from lidar_owl.losses import CrossEntropyFlat, resolve_loss

def test_cross_entropy_flat_matches_torch_functional():
    logits = torch.tensor(
        [[2.0, 0.1, -1.0], [0.1, 2.0, -1.0], [0.2, 0.1, 1.5]],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 1, 0], dtype=torch.long)
    weights = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float32)

    loss_mod = CrossEntropyFlat(ignore_index=0, class_weights=weights)
    loss_mod_val = loss_mod(logits, target)
    loss_ref_val = F.cross_entropy(logits, target, weight=weights, ignore_index=0)

    assert torch.isclose(loss_mod_val, loss_ref_val)


def test_ignore_index_masks_class_zero_points():
    logits_a = torch.tensor(
        [[8.0, -5.0], [0.2, 1.4], [7.0, -6.0], [0.1, 1.1]], dtype=torch.float32
    )
    logits_b = logits_a.clone()
    # Change ignored points massively -> loss must stay unchanged.
    logits_b[[0, 2]] = torch.tensor([[-100.0, 100.0], [100.0, -100.0]])
    target = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    loss_mod = CrossEntropyFlat(ignore_index=0)
    loss_a = loss_mod(logits_a, target)
    loss_b = loss_mod(logits_b, target)

    assert torch.isclose(loss_a, loss_b)


def test_out_of_range_target_raises_for_mapping_errors():
    logits = torch.randn(4, 3, dtype=torch.float32)  # classes are 0..2
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # class 3 invalid
    loss_mod = CrossEntropyFlat(ignore_index=0)

    with pytest.raises((RuntimeError, IndexError)):
        loss_mod(logits, target)


def test_ce_sanity_better_logits_give_lower_loss():
    target = torch.tensor([1, 2, 1, 0], dtype=torch.long)
    bad_logits = torch.tensor(
        [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        dtype=torch.float32,
    )
    good_logits = torch.tensor(
        [[0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    loss_mod = CrossEntropyFlat(ignore_index=-1)
    bad = loss_mod(bad_logits, target)
    good = loss_mod(good_logits, target)

    assert torch.isfinite(bad)
    assert torch.isfinite(good)
    assert good < bad


def test_ce_on_realistic_batch_shape_is_finite():
    # Typical semantic seg shape after flattening: (B * N, C)
    batch_size, num_points, num_classes = 2, 8, 4
    logits = torch.randn(batch_size * num_points, num_classes, dtype=torch.float32)
    target = torch.randint(0, num_classes, (batch_size * num_points,), dtype=torch.long)
    target[::5] = 0  # simulate ignored class prevalence

    loss_mod = CrossEntropyFlat(ignore_index=0)
    loss = loss_mod(logits, target)

    assert torch.isfinite(loss)
