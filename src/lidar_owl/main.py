import hydra
from omegaconf import OmegaConf, DictConfig

from ml3d_util import get_model, get_dataset
from pipelines import SemanticSegmentationExtended

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    # compact debug setup
    if cfg.get("debug"):
        cfg.dataset.training_split = ['08']
        cfg.dataset.validation_split = ['08']
        cfg.pipeline.max_epoch = 25
        cfg.pipeline.save_ckpt_freq = 0

    # set up model, dataset and semseg pipeline
    # config = OmegaConf.to_container(cfg)
    model = get_model(cfg.get("model", {}).get("name", {}))(**cfg.model)
    dataset = get_dataset(cfg.get("dataset", {}).get("name", {}))(**cfg.dataset)
    pipeline = SemanticSegmentationExtended(model, dataset, **cfg.pipeline)

    if cfg.mode == "train+eval":
        pipeline.run_train()
        pipeline.run_test()
    elif cfg.mode == "train":
        pipeline.run_train()
    elif cfg.mode == "eval":
        pipeline.run_test() 
    else:
        raise ValueError(f"Unknown mode '{cfg.mode}'")


if __name__ == "__main__":
    main()
