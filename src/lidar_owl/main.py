import hydra
import open3d as o3d
from omegaconf import DictConfig

from train import Trainer
from eval import Evaluator


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # keep console spam low
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)  # Error

    # compact debug setup
    if cfg.get("debug"):
        cfg.dataset.training_split = ['08']
        cfg.dataset.validation_split = ['08']
        cfg.pipeline.max_epoch = 10

    # TODO: checkpoint should never be loaded for train! for eval, open3d default (last one) is fine
    if cfg.mode in ("train_eval", "train+eval", "both"):
        semseg_trainer = Trainer(cfg)
        semseg_trainer.train()
        # reuse trained pipeline to directly run evaluation with trained weights
        semseg_trainer.pipeline.run_test()
    elif cfg.mode == "train":
        semseg_trainer = Trainer(cfg)
        semseg_trainer.train()
    elif cfg.mode == "eval":
        semseg_evaluator = Evaluator(cfg)
        semseg_evaluator.eval()
    else:
        raise ValueError(f"Unknown mode '{cfg.mode}'")


if __name__ == "__main__":
    main()
