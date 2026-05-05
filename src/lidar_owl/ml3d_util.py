import open3d.ml.torch as ml3d
import numpy as np
import torch

import open3d._ml3d.torch.dataloaders.default_batcher as default_batcher

from lidar_owl.datasets import SemanticKITTIFlat
from lidar_owl.models import RandLANetFlat

# dataset registry
def resolve_dataset(name: str):
    key = name.lower()
    if key in DATASET_REGISTRY:  # own dataset
        return DATASET_REGISTRY[key]
    if hasattr(ml3d.datasets, name):  # open3d-ml dataset
        return getattr(ml3d.datasets, name)
    raise KeyError(f"Unknown dataset '{name}'")
    
DATASET_REGISTRY = {
    "semantickittiflat": SemanticKITTIFlat,
}

# model registry
def resolve_model(name: str):
    key = name.lower()
    if key in MODEL_REGISTRY:  # own model
        return MODEL_REGISTRY[key]
    if hasattr(ml3d.models, name):  # open3d-ml model
        return getattr(ml3d.models, name)
    raise KeyError(f"Unknown model '{name}'")

MODEL_REGISTRY = {
    "randlanetflat": RandLANetFlat,
}


# monkeypatch for open3d-ml collate_fn to allow for num_workers > 0
def patch_open3d_default_batcher_resize_warning():
    if getattr(default_batcher, "_lidar_owl_resize_warning_patch", False):
        return

    original_default_collate = default_batcher.default_collate

    def default_collate_patched(batch):
        elem = batch[0]

        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                numel = sum(x.numel() for x in batch)
                if hasattr(elem, "_typed_storage"):
                    storage = elem._typed_storage()._new_shared(
                        numel,
                        device=elem.device,
                    )
                else:
                    storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *elem.size())
            return torch.stack(batch, 0, out=out)

        return original_default_collate(batch)

    default_batcher.default_collate = default_collate_patched
    default_batcher._lidar_owl_resize_warning_patch = True


# helper functions
def restore_prediction_labels(labels, ignored_label_inds):
    """Reinsert ignored labels into compact model predictions for visualization."""
    restored = np.array(labels, copy=True)
    # INVERSE of losses.filter_valid_semseg_labels(...):
    # Open3D trains/evaluates on compact class indices after ignored labels are removed
    # (e.g. SemanticKITTI predictions are 0..18, while dataset train IDs are 1..19).
    # For logging/export we must shift predictions back into the dataset label space.
    for ign_label in sorted(int(label) for label in ignored_label_inds):
        if ign_label >= 0:
            restored[restored >= ign_label] += 1
    return restored
