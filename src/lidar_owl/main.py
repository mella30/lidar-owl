import hydra
from omegaconf import DictConfig
from pathlib import Path
import shutil

from ml3d_util import get_model, get_dataset
from pipelines import SemanticSegmentationExtended

def _clean_checkpoints(cfg: DictConfig, model_name, dataset_name):
    main_log_dir = Path(cfg.pipeline.get("main_log_dir", "./logs"))
    target_dir = main_log_dir / f"{model_name}_{dataset_name}_torch"
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print(f"[clean] removed existing log dir: {target_dir}")

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    # compact debug setup
    if cfg.get("debug"):
        cfg.dataset.training_split = ['08']
        cfg.dataset.validation_split = ['08']
        cfg.pipeline.max_epoch = 25
        cfg.pipeline.save_ckpt_freq = 0

    model_name = cfg.get("model", {}).get("name", {})
    dataset_name = cfg.get("dataset", {}).get("name", {})
    if cfg.get("clean"):
        _clean_checkpoints(cfg, model_name, dataset_name)
    
    # set up model, dataset and pipeline
    model = get_model(model_name)(**cfg.model)
    dataset = get_dataset(dataset_name)(**cfg.dataset)
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
