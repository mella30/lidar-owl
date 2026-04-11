from datetime import datetime
from pathlib import Path

import open3d.ml.torch as ml3d
from open3d.ml.torch.modules import losses as ml3d_losses
from torch.utils.tensorboard import SummaryWriter

import torch

import lidar_owl.log as log
import lidar_owl.util as util
import lidar_owl.ml3d_util as ml3d_util
import lidar_owl.metrics as metrics

class SemanticSegmentationExtended(ml3d.pipelines.SemanticSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: implement early stopping
        # TODO: log git hash for each run

        # color palette for visu
        self.num_classes = self.dataset.num_classes
        self.num_trained_classes = self.model.cfg['num_classes']  # trained classes in model != available classes in dataset (incl. invalid)
        self.color_map = log.semkitti_cmap(self.num_classes)  # TODO: depends on dataset!
        self.ignored_label_inds = getattr(self.model.cfg, "ignored_label_inds", []) 
        self.class_names = log.compact_label_names_from_dataset(
            self.dataset,
            self.num_trained_classes,
            self.ignored_label_inds,
        )

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

        eval_log_dir = Path(self.cfg.eval_sum_dir) / timestamp
        writer = SummaryWriter(log_dir=str(eval_log_dir))
        writer.add_text("test/checkpoint_path", str(ckpt_path), 0)
        writer.add_scalar("test/checkpoint_epoch", int(ckpt_path.stem.split("_")[-1]), 0)
        return writer

    def _update_test_metric(self, inference_result, gt_labels):
        if not (gt_labels > 0).any():
            return

        # MUST use filter_valid_label here as well, otherwise metrics would compare
        # compact model predictions against non-compact dataset train IDs.
        valid_scores, valid_labels = ml3d_losses.filter_valid_label(
            torch.as_tensor(inference_result["predict_scores"], device=self.device),
            torch.as_tensor(gt_labels, device=self.device),
            num_classes=self.num_trained_classes,
            ignored_label_inds=self.ignored_label_inds,
            device=self.device,
        )
        self.metric_test.update(valid_scores, valid_labels)

    def save_logs(self, writer, epoch):
        # add visu of train / val preds
        stages = self.cfg.get("projection", {}).get("record_for", list(self.summary.keys()))

        for stage in stages:
            stage_summary = self.summary.get(stage, {})
            sem = stage_summary.get("semantic_segmentation")
            if not sem:
                continue

            xyz = util.tensor_to_np(sem.get("vertex_positions"))[0, :, :]
            gt = util.tensor_to_np(sem.get("vertex_gt_labels"))[0, :, :]
            pred = util.tensor_to_np(sem.get("vertex_predict_labels"))[0, :, :]
            pred = ml3d_util.restore_prediction_labels(pred, self.ignored_label_inds)
            visible_mask = (gt > 0).reshape(-1)

            gt_img = log.project(xyz, gt, self.color_map, visible_mask=visible_mask)
            pred_img = log.project(xyz, pred, self.color_map, visible_mask=visible_mask)

            if gt_img is not None:
                writer.add_image(f"{stage}/projection_gt", gt_img.transpose(2, 0, 1), epoch)
            if pred_img is not None:
                writer.add_image(f"{stage}/projection_pred", pred_img.transpose(2, 0, 1), epoch)

        super().save_logs(writer, epoch)

    def run_test(self, *args, **kwargs):

        # get checkpoint and tb writer
        ckpt_path = self._resolve_test_ckpt_path()
        self.model.cfg.ckpt_path = str(ckpt_path)
        self.load_ckpt(str(ckpt_path))

        self.metric_test = metrics.SemSegMetricExt(label_names=self.class_names)
        writer = self._create_test_writer(ckpt_path)

        test_dataset = self.dataset.get_split('test')

        for idx in range(len(test_dataset)):
            sample = test_dataset.get_data(idx)
            inference_result = self.run_inference(sample)
            
            pred_labels = ml3d_util.restore_prediction_labels(inference_result["predict_labels"], self.ignored_label_inds)
            gt_labels = sample["label"]
            self._update_test_metric(inference_result, gt_labels)
            
            log.log_projection_images(
                idx,
                sample["point"],
                pred_labels,
                gt_labels,
                self.color_map,
                writer,
            )

        if len(self.metric_test.acc()) > 0:
            writer.add_scalar("test/accuracy", self.metric_test.acc()[-1], len(test_dataset))
        if len(self.metric_test.iou()) > 0:
            writer.add_scalar("test/mIoU", self.metric_test.iou()[-1], len(test_dataset))

        writer.flush()
        writer.close()
