from typing import Any

import open3d.ml.torch as ml3d
from omegaconf import OmegaConf as OC

from ml3d_util import get_dataset, get_model, get_pipeline

# TODO: is that even neccessary here? incorporate into main experiment call
class Trainer:
    def __init__(self, config: dict[str, Any]):
        self.config = config  # incl model, dataset & pipeline

        self.model = get_model(self.config["model"]["name"])(**OC.to_container(config.model))
        self.dataset = get_dataset(self.config["dataset"]["name"])(**OC.to_container(config.dataset))
        self.pipeline = get_pipeline(self.config["pipeline"]["name"])(self.model, self.dataset, **OC.to_container(config.pipeline))

    def train(self):
        self.pipeline.run_train()
        print("Training completed.")
