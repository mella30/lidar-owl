from typing import Any

import open3d.ml.torch as ml3d
from omegaconf import OmegaConf

from ml3d_util import get_dataset, get_model, get_pipeline

# TODO: is that even neccessary here? incorporate into main experiment call
class Trainer:
    def __init__(self, config: dict[str, Any]):
        self.config = config  # incl model, dataset & pipeline

        # build model without loading a checkpoint unless a path is provided
        model_cfg = OmegaConf.to_container(config.model, resolve=True, enum_to_str=True)
        model_name = model_cfg.get("name")
        model_cls = get_model(model_name)
        ckpt_path = model_cfg.pop("ckpt_path", None)
        # remove meta keys not part of ctor signature
        model_cfg.pop("name", None)
        if ckpt_path:
            self.model = model_cls(ckpt_path=ckpt_path, **model_cfg)
        else:
            self.model = model_cls(**model_cfg)
        self.dataset = get_dataset(self.config["dataset"]["name"])(**config.dataset)
        self.pipeline = get_pipeline(self.config["pipeline"]["name"])(self.model, self.dataset, **config.pipeline)

    def train(self):
        self.pipeline.run_train()
        print("Training completed.")
