# semseg pipeline wrapper for various modifications
import numpy as np
import open3d.ml.torch as ml3d
import log


class SemanticSegmentationExtended(ml3d.pipelines.SemanticSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # color palette for visu
        self.color_map = log.semkitti_cmap(self.dataset.num_classes)  # TODO: depends on dataset!

    def save_logs(self, writer, epoch):
        # visu parameters
        visu_cfg = self.cfg.get('projection', {})
        # BEV images
        log.log_projection_images(epoch, self.summary, visu_cfg, self.color_map, writer)

        # class frequency summary (train IDs + mapped names)
        stage = "train"
        summary = self.summary.get(stage, {}).get("semantic_segmentation")
        if summary:
            num_classes = self.dataset.num_classes
            names = log.label_names_from_dataset(self.dataset, num_classes)

            preds = summary.get("vertex_predict_labels")
            if preds is not None:
                preds_np = np.asarray(preds).reshape(-1)
                counts_pred = np.bincount(preds_np, minlength=num_classes)
                lines_pred = [f"{i:02d} {names[i]}: {int(c)}" for i, c in enumerate(counts_pred) if c > 0]
                if lines_pred:
                    writer.add_text(f"{stage}/pred_class_hist", "\n".join(lines_pred), epoch)

            gt = summary.get("vertex_gt_labels")
            if gt is not None:
                gt_np = np.asarray(gt).reshape(-1)
                counts_gt = np.bincount(gt_np, minlength=num_classes)
                lines_gt = [f"{i:02d} {names[i]}: {int(c)}" for i, c in enumerate(counts_gt) if c > 0]
                if lines_gt:
                    writer.add_text(f"{stage}/gt_class_hist", "\n".join(lines_gt), epoch)

        # TODO: log calibration metrics

        # standard logs
        super().save_logs(writer, epoch)

    def run_train(self):
        return super().run_train()

    def run_test(self, *args, **kwargs):
        # Optionally return predictions and confidences from evaluation
        return_outputs = kwargs.pop("return_outputs", False)
        super().run_test(*args, **kwargs)
        if not return_outputs:
            return

        outputs = []
        for labels, scores in zip(self.ori_test_labels, self.ori_test_probs):
            scores_np = np.asarray(scores)
            outputs.append({
                "predict_labels": np.asarray(labels),
                "predict_scores": scores_np,
                "predict_confidences": scores_np.max(axis=1),
            })
        return outputs
