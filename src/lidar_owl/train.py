import open3d.ml.torch as ml3d

class Trainer:
    def __init__(self, config: dict[str, any]):
        self.config = config  # incl model, dataset & pipeline

        self.model = ml3d.models.RandLANet(**config.model)
        self.dataset = ml3d.datasets.SemanticKITTI(**config.dataset)
        self.pipeline = ml3d.pipelines.SemanticSegmentation(model=self.model, dataset=self.dataset, **config.pipeline)

    def train(self):
        self.pipeline.run_train()

        print("Training and testing completed.")
        return