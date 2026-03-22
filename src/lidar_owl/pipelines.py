import os, logging
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d.ml.torch as ml3d
from open3d.ml.torch.modules import metrics, losses
from torch.utils.tensorboard import SummaryWriter

import torch

import lidar_owl.log as log_owl
log = logging.getLogger(__name__)

class SemanticSegmentationExtended(ml3d.pipelines.SemanticSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: implement early stopping

        # color palette for visu
        self.color_map = log_owl.semkitti_cmap(self.dataset.num_classes)  # TODO: depends on dataset!

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

    def _create_test_writer(self, ckpt_path):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log.info("DEVICE : %s", self.device)
        log_file_path = os.path.join(self.cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : %s", log_file_path)
        log.addHandler(logging.FileHandler(log_file_path))

        eval_log_dir = Path(self.cfg.eval_sum_dir) / timestamp
        writer = SummaryWriter(log_dir=str(eval_log_dir))
        writer.add_text("test/checkpoint_path", str(ckpt_path), 0)
        writer.add_scalar("test/checkpoint_epoch", int(ckpt_path.stem.split("_")[-1]), 0)
        return writer

    def _update_test_metric(self, inference_result, gt_labels):
        if not (gt_labels > 0).any():
            return

        valid_scores, valid_labels = losses.filter_valid_label(
            torch.as_tensor(inference_result["predict_scores"], device=self.device),
            torch.as_tensor(gt_labels, device=self.device),
            self.model.cfg.num_classes,
            self.model.cfg.ignored_label_inds,
            self.device,
        )
        self.metric_test.update(valid_scores, valid_labels)
        log.info("Accuracy : %s", self.metric_test.acc())
        log.info("IoU : %s", self.metric_test.iou())

    def _build_test_output(self, cloud_id, inference_result):
        predict_scores = np.asarray(inference_result["predict_scores"])
        return {
            "cloud_id": cloud_id,
            "predict_labels": np.asarray(inference_result["predict_labels"]),
            "predict_scores": predict_scores,
            "predict_confidences": predict_scores.max(axis=1),
        }

    def save_logs(self, writer, epoch):
        ignored_label_inds = getattr(self.model.cfg, "ignored_label_inds", [])
        projection_cfg = self.cfg.get("projection", {})
        log_owl.log_projection_summary_images(
            epoch,
            self.summary,
            projection_cfg,
            self.color_map,
            writer,
            ignored_label_inds=ignored_label_inds,
        )
        super().save_logs(writer, epoch)

    def run_test(self, *args, **kwargs):
        return_outputs = kwargs.pop("return_outputs", False)

        ckpt_path = self._resolve_test_ckpt_path()
        self.model.cfg.ckpt_path = str(ckpt_path)
        self.load_ckpt(str(ckpt_path))

        self.metric_test = metrics.SemSegMetric()
        writer = self._create_test_writer(ckpt_path)

        test_dataset = self.dataset.get_split('test')
        ignored_label_inds = getattr(self.model.cfg, "ignored_label_inds", [])
        outputs = []

        log.info("Started testing")
        for idx in range(len(test_dataset)):
            sample = test_dataset.get_data(idx)
            inference_result = self.run_inference(sample)

            gt_labels = sample["label"]
            self._update_test_metric(inference_result, gt_labels)

            log_owl.log_projection_images(
                idx,
                sample["point"],
                inference_result["predict_labels"],
                gt_labels,
                self.color_map,
                writer,
                ignored_label_inds=ignored_label_inds,
            )

            self.dataset.save_test_result(inference_result, test_dataset.get_attr(idx))

            if return_outputs:
                outputs.append(self._build_test_output(idx, inference_result))

        if len(self.metric_test.acc()) > 0:
            writer.add_scalar("test/accuracy", self.metric_test.acc()[-1], len(test_dataset))
        if len(self.metric_test.iou()) > 0:
            writer.add_scalar("test/mIoU", self.metric_test.iou()[-1], len(test_dataset))
            log.info(
                "Overall Testing Accuracy : %s, mIoU : %s",
                self.metric_test.acc()[-1],
                self.metric_test.iou()[-1],
            )
        log.info("Finished testing")

        writer.flush()
        writer.close()

        if return_outputs:
            return outputs
