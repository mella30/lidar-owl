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
    if isinstance(loss_cfg, DictConfig):
        loss_cfg = loss_cfg.copy()
    if isinstance(loss_cfg, dict):
        name = loss_cfg.pop("name", None)
        if name and name.lower() in LOSS_REGISTRY:
            return LOSS_REGISTRY[name.lower()](**loss_cfg)
    # passthrough: already-instantiated loss or plain value handled by caller
    return loss_cfg
