from typing import Any

import open3d.ml.torch as ml3d
from omegaconf import OmegaConf
from ml3d_util import get_model

class Evaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config  # incl model, dataset & pipeline

        model_cfg = OmegaConf.to_container(config.model, resolve=True, enum_to_str=True)
        model_name = model_cfg.get("name")
        model_cls = get_model(model_name)
        ckpt_path = model_cfg.pop("ckpt_path", None)
        model_cfg.pop("name", None)
        if ckpt_path:
            self.model = model_cls(ckpt_path=ckpt_path, **model_cfg)
        else:
            self.model = model_cls(**model_cfg)
        self.dataset = ml3d.datasets.SemanticKITTI(**config.dataset)
        self.pipeline = ml3d.pipelines.SemanticSegmentation(
            model=self.model,
            dataset=self.dataset,
            **config.pipeline,
        )

    def eval(self):
        self.pipeline.run_test()  # or run_inference?
        print("Evaluation completed.")
