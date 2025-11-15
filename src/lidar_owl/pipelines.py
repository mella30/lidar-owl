# semseg pipeline wrapper for various modifications
import numpy as np
import open3d.ml.torch as ml3d
import log, util

class SemanticSegmentationExtended(ml3d.pipelines.SemanticSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        label_map = self.dataset.get_label_to_names()
        self._palette = log.semkitti_palette(len(label_map))
    
    def _log_projection_images(self, writer, epoch):
        # visualizes GT and preds per epoch
        cfg = self.cfg.get('projection', {})
        if not cfg.get('enabled', True):
            return
        stages = cfg.get('record_for', list(self.summary.keys()))
        size = tuple(cfg.get('image_size', [512, 512]))
        axes = tuple(cfg.get('axes', [0, 1]))
        depth_axis = cfg.get('depth_axis', 2)

        # TODO: heuristic for which PCs to log (not random)
        for stage in stages:
            stage_summary = self.summary.get(stage, {})
            sem = stage_summary.get('semantic_segmentation')
            if not sem:
                continue
            xyz = util.tensor_to_np(sem.get('vertex_positions'))
            gt = util.tensor_to_np(sem.get('vertex_gt_labels'))
            pred = util.tensor_to_np(sem.get('vertex_predict_labels'))
            if xyz is None or pred is None:
                continue
            xyz = np.squeeze(xyz, axis=0)
            gt = np.squeeze(gt, axis=0) if gt is not None else None
            pred = np.squeeze(pred, axis=0)

            gt_img = log.project(xyz, gt, self._palette, size, axes, depth_axis) if gt is not None else None
            pred_img = log.project(xyz, pred, self._palette, size, axes, depth_axis)
            if gt_img is not None:
                writer.add_image(f"{stage}/projection_gt",
                                 gt_img.transpose(2, 0, 1), epoch)
            if pred_img is not None:
                writer.add_image(f"{stage}/projection_pred",
                                 pred_img.transpose(2, 0, 1), epoch)

    def save_logs(self, writer, epoch):
        self._log_projection_images(writer, epoch)
        super().save_logs(writer, epoch)
