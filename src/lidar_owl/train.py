from typing import Any

import open3d.ml.torch as ml3d

from ml3d_util import get_dataset, get_model, get_pipeline

# TODO: is that even neccessary here? incorporate into main experiment call
class Trainer:
    def __init__(self, config: dict[str, Any]):
        self.config = config  # incl model, dataset & pipeline

        self.model = get_model(self.config["model"]["name"])(**config.model)
        self.dataset = get_dataset(self.config["dataset"]["name"])(**config.dataset)
        self.pipeline = get_pipeline(self.config["pipeline"]["name"])(self.model, self.dataset, **config.pipeline)

    def train(self):
        self.pipeline.run_train()
        print("Training completed.")
