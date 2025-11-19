import hydra
import open3d as o3d
import omegaconf as OC

from train import Trainer
from eval import Evaluator


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: OC.DictConfig):
    # compact debug setup
    if cfg.get("debug"):
        cfg.dataset.training_split = ['08']
        cfg.dataset.validation_split = ['08']
        cfg.pipeline.max_epoch = 25
        cfg.pipeline.save_ckpt_freq = 0

    if cfg.mode == "train+eval":
        semseg_trainer = Trainer(cfg)
        semseg_trainer.train()
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
