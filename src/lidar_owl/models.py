# TODO: hierarchical models, derive from ml3d base model 
# (also consider class remapping in dataset)
# TODO: uncertainty models
# TODO: anomaly models

import open3d.ml.torch as ml3d
from omegaconf import OmegaConf

from lidar_owl.losses import resolve_loss

# open3d-ml model wrapper
class RandLANetFlat(ml3d.models.RandLANet):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        # ensure augment block is a plain dict (Open3D mutates it)
        augment_cfg = kwargs["augment"]
        if augment_cfg is not None:
            kwargs["augment"] = OmegaConf.to_container(augment_cfg, resolve=True)
        # reset name for later calls
        kwargs["name"] = "RandLANet"

        # resolve configured loss before Open3D model init
        resolved_loss = resolve_loss(kwargs["loss"])
        if resolved_loss is not None:
            kwargs["loss"] = resolved_loss

        super().__init__(*args, **kwargs)

    def transform(self, data, attr, min_possibility_idx=None):
        inputs = super().transform(data, attr, min_possibility_idx)

        # ml3d bug for visu of randlanet: mismatch in key names
        if isinstance(inputs, dict) and 'xyz' not in inputs:
            coords = inputs['coords']
            if coords is not None:
                inputs['xyz'] = coords
        return inputs
