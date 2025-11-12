import open3d
import open3d.ml.torch as ml3d

# dataset wrapper
class SemanticKITTISplitFlat(open3d._ml3d.datasets.semantickitti.SemanticKITTISplit):
    def get_data(self, idx):
        sample = super().get_data(idx)
        sample["feat"] = None 
        return sample

class SemanticKITTIFlat(ml3d.datasets.SemanticKITTI):
    def get_split(self, split):
        return SemanticKITTISplitFlat(self, split=split)


# dataset registry
def get_dataset(name: str):
    key = name.lower()
    if key in DATASET_REGISTRY:  # own dataset
        return DATASET_REGISTRY[key]
    if hasattr(ml3d.datasets, name):  # open3d-ml dataset
        return getattr(ml3d.datasets, name)
    raise KeyError(f"Unknown dataset '{name}'")
    
DATASET_REGISTRY = {
    "semantickittiflat": SemanticKITTIFlat,
}


# model wrapper 
class RandLANetFlat(ml3d.models.RandLANet):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)

        # reset name for later calls
        kwargs["name"] = "RandLANet"
        super().__init__(*args, **kwargs)

    def transform(self, data, attr, min_possibility_idx=None):
        inputs = super().transform(data, attr, min_possibility_idx)
        if isinstance(inputs, dict) and 'xyz' not in inputs:
            coords = inputs.get('coords')
            if coords is not None:
                inputs['xyz'] = coords
        return inputs

# model registry
def get_model(name: str):
    key = name.lower()
    if key in MODEL_REGISTRY:  # own model
        return MODEL_REGISTRY[key]
    if hasattr(ml3d.models, name):  # open3d-ml model
        return getattr(ml3d.models, name)
    raise KeyError(f"Unknown model '{name}'")

MODEL_REGISTRY = {
    "randlanetflat": RandLANetFlat,
}