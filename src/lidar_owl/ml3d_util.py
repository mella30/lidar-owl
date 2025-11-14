import open3d.ml.torch as ml3d

from datasets import SemanticKITTIFlat
from models import RandLANetFlat
from pipelines import SemanticSegmentationExtended

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

# pipeline registry
def get_pipeline(name: str):
    key = name.lower()
    if key in PIPELINE_REGISTRY:  # own model
        return PIPELINE_REGISTRY[key]
    if hasattr(ml3d.pipelines, name):  # open3d-ml model
        return getattr(ml3d.pipelines, name)
    raise KeyError(f"Unknown pipeline '{name}'")

PIPELINE_REGISTRY = {
    "semsegext": SemanticSegmentationExtended,
}