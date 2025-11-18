from typing import Any

import open3d.ml.torch as ml3d

class Evaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config  # incl model, dataset & pipeline

        self.model = ml3d.models.RandLANet(**config.model)
        self.dataset = ml3d.datasets.SemanticKITTI(**config.dataset)
        self.pipeline = ml3d.pipelines.SemanticSegmentation(
            model=self.model,
            dataset=self.dataset,
            **config.pipeline,
        )

    def eval(self):
        self.pipeline.run_test()  # or run_inference?
        print("Evaluation completed.")
