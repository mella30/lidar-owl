import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import shutil

from lidar_owl.ml3d_util import resolve_model, resolve_dataset
from lidar_owl.pipelines import SemanticSegmentationExtended

def _clean_checkpoints(cfg: DictConfig, model_name, dataset_name):
    main_log_dir = Path(cfg.pipeline["main_log_dir"])
    target_dir = main_log_dir / f"{model_name}_{dataset_name}_torch"
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print(f"[clean] removed existing log dir: {target_dir}")
    cache_dir = Path(cfg.dataset["cache_dir"])
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"[clean] removed cache dir: {cache_dir}")

# TODO: remove in final version
def _open3d_ce_class_weights(num_per_class):
    num_per_class = np.asarray(num_per_class, dtype=np.float32)
    weight = num_per_class / float(num_per_class.sum())
    ce_label_weight = 1.0 / (weight + 0.02)
    return ce_label_weight.tolist()

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    model_name = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]
    if cfg.get("clean", False):
        _clean_checkpoints(cfg, model_name, dataset_name)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    # forward class weights from dataset config to loss config if requested
    # TODO: remove in final version?
    loss_cfg = model_cfg.get("loss") if isinstance(model_cfg, dict) else None
    if isinstance(loss_cfg, dict):
        class_weights_cfg = loss_cfg.get("class_weights", None)
        if class_weights_cfg is True:
            dataset_weights = dataset_cfg.get("class_weights")
            if dataset_weights is None:
                raise KeyError(
                    "loss.class_weights is true, but dataset.class_weights is not configured."
                )
            loss_cfg["class_weights"] = _open3d_ce_class_weights(dataset_weights)
        elif class_weights_cfg is False:
            loss_cfg["class_weights"] = None

    # set up model, dataset and pipeline
    model = resolve_model(model_name)(**model_cfg)
    dataset = resolve_dataset(dataset_name)(**cfg.dataset)
    pipeline = SemanticSegmentationExtended(model, dataset, **cfg.pipeline)

    if cfg.mode == "train+eval":
        pipeline.run_train()
        return pipeline.run_test()
    elif cfg.mode == "train":
        pipeline.run_train()
    elif cfg.mode == "eval":
        return pipeline.run_test()
    else:
        raise ValueError(f"Unknown mode '{cfg.mode}'")


if __name__ == "__main__":
    main()
