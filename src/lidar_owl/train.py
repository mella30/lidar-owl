import open3d.ml.torch as ml3d

from ml3d_util import get_dataset, get_model

class Trainer:
    def __init__(self, config: dict[str, any]):
        self.config = config  # incl model, dataset & pipeline

        self.model = get_model(self.config["model"]["name"])(**config.model)
        self.dataset = get_dataset(self.config["dataset"]["name"])(**config.dataset)

        self.pipeline = ml3d.pipelines.SemanticSegmentation(
            model=self.model,
            dataset=self.dataset,
            **config.pipeline,
        )

    def train(self):
        self.pipeline.run_train()
        print("Training completed.")
