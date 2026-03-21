from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
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
        # load the latest checkpoint in given path
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

        # load model and create new eval tb
        ckpt_path = self._resolve_test_ckpt_path()
        self.model.cfg.ckpt_path = str(ckpt_path)
        self.load_ckpt(str(ckpt_path))

        # TODO: naming!
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_sum_dir = Path(self.cfg.eval_sum_dir) / timestamp
        writer = SummaryWriter(log_dir=str(eval_sum_dir))

        # loop over test set and run inference 
        test_dataset = self.dataset.get_split("test")

        for i in range(len(test_dataset)):
            print(i)
            sample = test_dataset.get_data(i)
            model_results = self.run_inference(sample)

            preds = model_results['predict_labels']
            confs = model_results['predict_scores']
            labels = sample['label']

            log.log_projection_images(i, sample['point'], preds, labels, self.color_map, writer, self.model.cfg["ignored_label_inds"])

    
