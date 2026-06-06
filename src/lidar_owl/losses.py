# TODO: hierarchical losses, metric learning

import logging

import torch
import torch.nn.functional as F

class CrossEntropyFlat(torch.nn.Module):
    # Cross-entropy on compact semantic-segmentation labels.
    # Important:
    # - ignored points must already be removed before calling this loss
    # - label compaction (`learned/train IDs -> compact IDs`) happens in the model's `get_loss()` adapter, not in this class

    def __init__(self, ignore_index=-1, class_weights=None, num_classes=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = (
            None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        )
        self.num_classes = num_classes

    def forward(self, logits, target):
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device)
        return F.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
        )


LOSS_REGISTRY = {
    "crossentropyflat": CrossEntropyFlat,
}


def resolve_loss(loss_cfg, num_classes=None):
    """
    Resolve the loss function based on the configuration.
    Allowed are only the losses defined in LOSS_REGISTRY and open3dml's SemSegLoss as default fallback.
    """
    if loss_cfg is None:
        logging.info("No loss specified. Fallback to Open3D-ML default SemSegLoss...")
        return None  # default open3dml's SemSegLoss will be used
    loss_name = str(loss_cfg.pop("name"))

    # try to load from own loss registry & return
    loss_class = LOSS_REGISTRY.get(loss_name.lower())
    if loss_class is not None:
        # set given umber of classes as default (defined in dataset config)
        loss_cfg.setdefault("num_classes", num_classes)
        return loss_class(**loss_cfg)

    logging.info(f"Unknown loss '{loss_name}'. Available losses: {', '.join(sorted(LOSS_REGISTRY))}")
    logging.info("Fallback to Open3D-ML default SemSegLoss...")
    return None
