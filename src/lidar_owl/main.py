import hydra
from omegaconf import DictConfig
from pathlib import Path
import shutil

from lidar_owl.ml3d_util import resolve_model, resolve_dataset
from lidar_owl.pipelines import SemanticSegmentationExtended

def _clean_checkpoints(cfg: DictConfig, model_name, dataset_name):
    main_log_dir = Path(cfg.pipeline.get("main_log_dir", "./logs"))
    target_dir = main_log_dir / f"{model_name}_{dataset_name}_torch"
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print(f"[clean] removed existing log dir: {target_dir}")
    cache_dir = Path(cfg.dataset.get("cache_dir", "./logs/cache"))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"[clean] removed cache dir: {cache_dir}")

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):


    model_name = cfg.get("model", {}).get("name", {})
    dataset_name = cfg.get("dataset", {}).get("name", {})
    if cfg.get("clean"):
        _clean_checkpoints(cfg, model_name, dataset_name)
    
    # set up model, dataset and pipeline
    model = resolve_model(model_name)(**cfg.model)
    dataset = resolve_dataset(dataset_name)(**cfg.dataset)
    pipeline = SemanticSegmentationExtended(model, dataset, **cfg.pipeline)
    return_outputs = cfg.pipeline.get("return_outputs", False)

    if cfg.mode == "train+test":
        pipeline.run_train()
        return pipeline.run_test(return_outputs=return_outputs)
    elif cfg.mode == "train":
        pipeline.run_train()
    elif cfg.mode == "test":
        return pipeline.run_test(return_outputs=return_outputs)
    else:
        raise ValueError(f"Unknown mode '{cfg.mode}'")


if __name__ == "__main__":
    main()
