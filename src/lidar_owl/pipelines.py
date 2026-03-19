from datetime import datetime
from pathlib import Path

import numpy as np
import open3d.ml.torch as ml3d
from torch.utils.tensorboard import SummaryWriter

import lidar_owl.log as log

class SemanticSegmentationExtended(ml3d.pipelines.SemanticSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: implement early stopping

        # color palette for visu
        self.color_map = log.semkitti_cmap(self.dataset.num_classes)  # TODO: depends on dataset!

    def _resolve_test_ckpt_path(self):
        ckpt_path = getattr(self.model.cfg, "ckpt_path", None)
        if ckpt_path:
            return Path(ckpt_path)

        model_name = self.model.__class__.__name__
        dataset_name = self.dataset.__class__.__name__
        ckpt_dir = Path(self.cfg.main_log_dir) / f"{model_name}_{dataset_name}_torch" / "checkpoint"
        ckpt_paths = sorted(ckpt_dir.glob("ckpt_*.pth"))
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        return ckpt_paths[-1]

    def run_test(self, *args, **kwargs):  # / TODO: see also update_tests
        ckpt_path = self._resolve_test_ckpt_path()
        self.model.cfg.ckpt_path = str(ckpt_path)
        self.load_ckpt(str(ckpt_path))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_log_dir = Path(self.cfg.eval_log_dir) / timestamp
        writer = SummaryWriter(log_dir=str(eval_log_dir))
        writer.add_text("test/checkpoint_path", str(ckpt_path), 0)
        writer.add_scalar("test/checkpoint_epoch", int(ckpt_path.stem.split("_")[-1]), 0)
        writer.flush()
        writer.close()

        return {
            "checkpoint_path": str(ckpt_path),
            "tensorboard_dir": str(eval_log_dir),
        }
    
