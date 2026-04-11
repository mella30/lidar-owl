# TODO: hierarchical losses, metric learning

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


class CrossEntropyFlat(torch.nn.Module):
    # Cross-entropy on compact semantic-segmentation labels.
    # Important:
    # - ignored points must already be removed before calling this loss
    # - label compaction (`learned/train IDs -> compact IDs`) happens in the model's `get_loss()` adapter, not in this class

    def __init__(self, ignore_index=-1, class_weights=None, num_classes=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.num_classes = num_classes

    def forward(self, logits, target):
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights.to(device=logits.device),
            ignore_index=self.ignore_index,
        )


LOSS_REGISTRY = {
    "crossentropyflat": CrossEntropyFlat,
}


def resolve_loss(loss_cfg, num_classes=None):
    if loss_cfg is None:
        return None

    if isinstance(loss_cfg, torch.nn.Module):
        return loss_cfg

    if isinstance(loss_cfg, DictConfig):
        loss_cfg = OmegaConf.to_container(loss_cfg, resolve=True)

    if isinstance(loss_cfg, dict):
        cfg = dict(loss_cfg)
        name = cfg.pop("name", None)
        if not name:
            raise KeyError("Loss config must contain a 'name' field.")
        key = name.lower()
        if key not in LOSS_REGISTRY:
            available = ", ".join(sorted(LOSS_REGISTRY.keys()))
            raise KeyError(f"Unknown loss '{name}'. Available losses: {available}")
        cfg.setdefault("num_classes", num_classes)
        return LOSS_REGISTRY[key](**cfg)

    raise TypeError(
        "Unsupported loss config type. Expected DictConfig/dict with 'name' or torch.nn.Module "
        f"instance, got {type(loss_cfg).__name__}."
    )
