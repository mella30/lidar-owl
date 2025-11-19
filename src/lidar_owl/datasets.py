# TODO: class mapping for hierarchical trainings
# TODO: mask managing (via yaml)
# TODO: carla dataset

import open3d
import open3d.ml.torch as ml3d

from omegaconf import OmegaConf as OC

# open3d-ml dataset wrapper
class SemanticKITTISplitFlat(open3d._ml3d.datasets.semantickitti.SemanticKITTISplit):
    def get_data(self, idx):
        sample = super().get_data(idx)
        # remove intensity from feature channel (should not be used for training since it's quite uncalibrated)
        sample["feat"] = None 
        return sample

class SemanticKITTIFlat(ml3d.datasets.SemanticKITTI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_split(self, split):
        return SemanticKITTISplitFlat(self, split=split)