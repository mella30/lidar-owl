# TODO: hierarchical models, derive from ml3d base model 
# (also consider class remapping in dataset)
# TODO: uncertainty models
# TODO: anomaly models

import open3d.ml.torch as ml3d
import numpy as np
from omegaconf import OmegaConf
from open3d._ml3d.datasets.utils import DataProcessing

from lidar_owl.losses import resolve_loss

# open3d-ml model wrapper
class RandLANetFlat(ml3d.models.RandLANet):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        # TODO: still, a random crop of the PC is selected for train & eval. that should NOT be the case!
        # ensure augment block is a plain dict (Open3D mutates it)
        augment_cfg = kwargs.get("augment", None)
        if augment_cfg is not None:
            kwargs["augment"] = OmegaConf.to_container(augment_cfg, resolve=True)
        # reset name for later calls
        kwargs["name"] = "RandLANet"

        # resolve configured loss before Open3D model init
        resolved_loss = resolve_loss(kwargs.get("loss", "CrossEntropyLoss"))  # TODO: check if default would work
        if resolved_loss is not None:
            kwargs["loss"] = resolved_loss

        super().__init__(*args, **kwargs)

    def _deterministic_debug_crop(self, pc, feat, label, tree):
        if self.cfg.get("debug_use_full_frame", False):
            selected_idxs = np.arange(pc.shape[0], dtype=np.int64)
            center_point = np.zeros((1, pc.shape[1]), dtype=pc.dtype)
            return pc.copy(), feat.copy() if feat is not None else None, label.copy(), selected_idxs

        num_points = min(self.cfg.num_points, pc.shape[0])
        pick_idx = int(self.cfg.get("debug_fixed_pick_idx", 0))
        pick_idx = max(0, min(pick_idx, pc.shape[0] - 1))
        center_point = pc[pick_idx, :].reshape(1, -1)

        if pc.shape[0] <= num_points:
            selected_idxs = np.arange(pc.shape[0], dtype=np.int64)
        else:
            selected_idxs = tree.query(center_point, k=num_points)[1][0].astype(np.int64)
            selected_idxs.sort()

        cropped_pc = pc[selected_idxs] - center_point
        cropped_feat = feat[selected_idxs] if feat is not None else None
        cropped_label = label[selected_idxs]
        return cropped_pc, cropped_feat, cropped_label, selected_idxs

    def transform(self, data, attr, min_possibility_idx=None):
        if self.cfg.get("debug_fixed_crop", False):
            inputs = dict()

            pc = data["point"].copy()
            label = data["label"].copy()
            feat = data["feat"].copy() if data["feat"] is not None else None
            tree = data["search_tree"]

            pc, feat, label, selected_idxs = self._deterministic_debug_crop(
                pc, feat, label, tree
            )

            augment_cfg = self.cfg.get("augment", None).copy()
            val_augment_cfg = {}
            if "recenter" in augment_cfg:
                val_augment_cfg["recenter"] = augment_cfg.pop("recenter")
            if "normalize" in augment_cfg:
                val_augment_cfg["normalize"] = augment_cfg.pop("normalize")

            # In debug overfit mode keep transforms deterministic across epochs.
            pc, feat, label = self.augmenter.augment(
                pc, feat, label, val_augment_cfg, seed=np.random.default_rng(0)
            )

            if feat is None:
                feat = pc.copy()
            else:
                feat = np.concatenate([pc, feat], axis=1)

            if self.cfg.in_channels != feat.shape[1]:
                raise RuntimeError(
                    "Wrong feature dimension, please update in_channels(3 + feature_dimension) in config"
                )

            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(self.cfg.num_layers):
                neighbour_idx = DataProcessing.knn_search(pc, pc, self.cfg.num_neighbors)
                sub_points = pc[: pc.shape[0] // self.cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[: pc.shape[0] // self.cfg.sub_sampling_ratio[i], :]
                up_i = DataProcessing.knn_search(sub_points, pc, 1)
                input_points.append(pc)
                input_neighbors.append(neighbour_idx.astype(np.int64))
                input_pools.append(pool_i.astype(np.int64))
                input_up_samples.append(up_i.astype(np.int64))
                pc = sub_points

            inputs["coords"] = input_points
            inputs["neighbor_indices"] = input_neighbors
            inputs["sub_idx"] = input_pools
            inputs["interp_idx"] = input_up_samples
            inputs["features"] = feat
            inputs["point_inds"] = selected_idxs
            inputs["labels"] = label.astype(np.int64)
        else:
            inputs = super().transform(data, attr, min_possibility_idx)

        # ml3d bug for visu of randlanet: mismatch in key names
        if isinstance(inputs, dict) and 'xyz' not in inputs:
            coords = inputs.get('coords', None)
            if coords is not None:
                inputs['xyz'] = coords
        return inputs
