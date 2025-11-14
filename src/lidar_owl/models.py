# TODO: hierarchical models, derive from ml3d base model 
# (also consider class remapping in dataset)
# TODO: uncertainty models
# TODO: anomaly models

import open3d.ml.torch as ml3d

# open3d-ml model wrapper
class RandLANetFlat(ml3d.models.RandLANet):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)

        # reset name for later calls
        kwargs["name"] = "RandLANet"
        super().__init__(*args, **kwargs)

    def transform(self, data, attr, min_possibility_idx=None):
        inputs = super().transform(data, attr, min_possibility_idx)
        # ml3d bug for visu of randlanet: mismatch in key names
        if isinstance(inputs, dict) and 'xyz' not in inputs:
            coords = inputs.get('coords')
            if coords is not None:
                inputs['xyz'] = coords
        return inputs