# TODO: hierarchical models, derive from ml3d base model 
# (also consider class remapping in dataset)
# TODO: uncertainty models
# TODO: anomaly models

import open3d.ml.torch as ml3d
from omegaconf import OmegaConf

from open3d.ml.torch.modules import losses as ml3d_losses
from lidar_owl.losses import resolve_loss

class BaseFlatAdapter:
    # Mixin to adapt Open3D-ML models to use our configured loss wrappers instead of their hardcoded CE path.
    def get_loss(self, Loss, results, inputs, device):
        labels = inputs["data"]["labels"]
        scores, labels =  ml3d_losses.filter_valid_label(
            results,
            labels,
            num_classes=self.cfg.num_classes,
            ignored_label_inds=self.cfg.ignored_label_inds,
            device=device,
        )
        configured_loss = getattr(self, "custom_loss", None)
        if callable(configured_loss):
            loss = configured_loss(scores, labels)
        else:
            # Fallback to Open3D-ML's default semseg loss object when no custom
            # callable loss was attached to the model.
            loss = Loss.weighted_CrossEntropyLoss(scores, labels)
        return loss, labels, scores

# open3d-ml model wrapper
class RandLANetFlat(BaseFlatAdapter, ml3d.models.RandLANet):  # get_loss mixin must be before ml3d model in MRO
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        # reset name for later calls
        kwargs["name"] = "RandLANet"

        # resolve configured loss before Open3D model init
        resolved_loss = resolve_loss(
            kwargs["loss"],
            num_classes=kwargs.get("num_classes"),
        )
        if resolved_loss is not None:
            kwargs["loss"] = resolved_loss

        super().__init__(*args, **kwargs)
        self.custom_loss = resolved_loss

    def transform(self, data, attr, min_possibility_idx=None):
        inputs = super().transform(data, attr, min_possibility_idx)

        # ml3d bug for visu of randlanet: mismatch in key names
        if isinstance(inputs, dict) and 'xyz' not in inputs:
            coords = inputs['coords']
            if coords is not None:
                inputs['xyz'] = coords
        return inputs
