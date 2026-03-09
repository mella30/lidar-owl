# TODO: hierarchical losses, metric learning

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


class CrossEntropyFlat(torch.nn.Module):
    def __init__(self, ignore_index=-1, class_weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits, target):
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
    

LOSS_REGISTRY = {
    "crossentropyflat": CrossEntropyFlat,
}


def resolve_loss(loss_cfg):
    if loss_cfg is None:
        return None

    if isinstance(loss_cfg, nn.Module):
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
        return LOSS_REGISTRY[key](**cfg)

    raise TypeError(
        "Unsupported loss config type. Expected DictConfig/dict with 'name' or torch.nn.Module "
        f"instance, got {type(loss_cfg).__name__}."
    )
