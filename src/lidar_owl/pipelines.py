# semseg pipeline wrapper for various modifications
import numpy as np
import open3d.ml.torch as ml3d

import lidar_owl.log as log

class SemanticSegmentationExtended(ml3d.pipelines.SemanticSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: implement early stopping

        # color palette for visu
        self.color_map = log.semkitti_cmap(self.dataset.num_classes)  # TODO: depends on dataset!

    def _get_metric_obj(self, stage):
        # helper function to access stage metrics
        if stage == "train":
            return getattr(self, "metric_train", None)
        if stage == "val":
            return getattr(self, "metric_val", None)
        if stage == "test":
            return getattr(self, "metric_test", None)
        return None

    def save_logs(self, writer, epoch):
        # train logger only (BEV visu extension for sanity-checking)
        visu_cfg = self.cfg.get('projection', {})

        # BEV images
        ignored_label_inds = getattr(self.model.cfg, "ignored_label_inds", [])
        log.log_projection_images(
            epoch,
            self.summary,
            visu_cfg,
            self.color_map,
            writer,
            ignored_label_inds=ignored_label_inds,
        )

        # standard logs
        super().save_logs(writer, epoch)

    def run_test(self, *args, **kwargs):  # / TODO: see also update_tests
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

        # TODO: log calibration metrics & 3d visu

        return outputs
    

    # also available: run_train, run_inference (preds only) / get_3d_summary()