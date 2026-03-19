# TODO: visu, runtime
import numpy as np
import yaml
from pathlib import Path
import open3d

import lidar_owl.util as util


def restore_prediction_labels(labels, ignored_label_inds):
    """Reinsert ignored labels into compact model predictions for visualization."""
    restored = np.array(labels, copy=True)
    # Open3D trains on compact class indices after ignored labels are removed
    # (e.g. SemanticKITTI predictions are 0..18, while dataset train IDs are 1..19).
    # For logging/export we need to shift predictions back into the dataset label space.
    for ign_label in sorted(int(label) for label in ignored_label_inds):
        if ign_label >= 0:
            restored[restored >= ign_label] += 1
    return restored


def semkitti_cmap(num_classes: int) -> np.ndarray:
    # gets semantickitti colors from open3d lib 
    resource = Path(open3d._ml3d.__file__).parent / "datasets" / "_resources" / "semantic-kitti.yaml"
    data = yaml.safe_load(resource.read_text())
    # remap colors from preds to original semnantickitti colors
    color_map = {int(k): np.array(v, dtype=np.float32) / 255.0
                for k, v in data["color_map"].items()} 
    inv_map = {int(k): int(v) for k, v in data["learning_map_inv"].items()}
    palette = np.zeros((num_classes, 3), dtype=np.float32)
    for train_id in range(num_classes):
        raw_id = inv_map.get(train_id, 0)  # fallback: ignored index
        palette[train_id] = color_map.get(raw_id, np.ones(3, dtype=np.float32))
    return palette

def semkitti_train_id_to_name(num_classes: int) -> list[str]:
    """Map train IDs (after learning_map) to human readable SemanticKITTI names."""
    resource = Path(open3d._ml3d.__file__).parent / "datasets" / "_resources" / "semantic-kitti.yaml"
    data = yaml.safe_load(resource.read_text())
    inv_map = {int(k): int(v) for k, v in data["learning_map_inv"].items()}
    labels = {int(k): v for k, v in data["labels"].items()}
    names = []
    for train_id in range(num_classes):
        raw_id = inv_map.get(train_id, -1)  # fallback: ignored index
        names.append(labels.get(raw_id, f"raw_{raw_id}"))
    return names

def label_names_from_dataset(dataset, num_classes: int) -> list[str]:
    """Prefer dataset-provided label_to_names (train IDs); fallback to SemanticKITTI mapping."""
    names_map = getattr(dataset, "label_to_names", None)
    if isinstance(names_map, dict) and names_map:
        return [names_map.get(i, f"class_{i}") for i in range(num_classes)]
    return semkitti_train_id_to_name(num_classes)


def compact_label_names_from_dataset(dataset, num_classes: int, ignored_label_inds) -> list[str]:
    """Names for the compact model label space after ignored labels are removed."""
    full_names = label_names_from_dataset(dataset, num_classes + len(ignored_label_inds))
    ignored = set(int(label) for label in ignored_label_inds if int(label) >= 0)
    return [name for idx, name in enumerate(full_names) if idx not in ignored][:num_classes]


def project(points, labels, palette, size=(512, 512), axes=(0, 1), depth_axis=2, visible_mask=None):
    # BEV projection -> TODO: sensor view projection?
    if points.size == 0 or labels is None:
        return None
    w, h = size
    coords = points[:, axes]
    mins = coords.min(0)
    spans = np.maximum(coords.max(0) - mins, 1e-6)
    norm = (coords - mins) / spans
    pix = np.clip((norm * np.array([w - 1, h - 1])).round().astype(int),
                  [0, 0], [w - 1, h - 1])
    depth = points[:, depth_axis] if depth_axis is not None else np.arange(len(points))

    if visible_mask is None:
        mask = (labels > 0).reshape(-1)
    else:
        mask = np.asarray(visible_mask, dtype=bool).reshape(-1)
    pix = pix[mask]
    depth = depth[mask]
    lbs = labels.reshape(-1)[mask]
    if pix.size == 0:
        return None
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    zbuf = np.full((h, w), -np.inf)
    for (x, y), z, label in zip(pix, depth, lbs):
        if z >= zbuf[y, x]:
            zbuf[y, x] = z
            canvas[y, x] = palette[label]
    return canvas

def log_projection_images(epoch, stage_summary, cfg, palette, writer, ignored_label_inds=()):
    # TODO: clean that mess up!!!
    # TODO: currently only works for train (does not log val or test, but should)
    # visualizes GT and preds per epoch
    size = tuple(cfg.get('image_size', [512, 512]))
    axes = tuple(cfg.get('axes', [0, 1])) 
    depth_axis = cfg.get('depth_axis', 2)

    # TODO: heuristic for which PCs to log (not random)
    sem = stage_summary.get('semantic_segmentation')

    # take only first sample from batched data
    xyz = util.tensor_to_np(sem.get('vertex_positions'))[0, :, :]
    gt = util.tensor_to_np(sem.get('vertex_gt_labels'))[0, :, :]
    pred = util.tensor_to_np(sem.get('vertex_predict_labels'))[0, :, :]
    pred = restore_prediction_labels(pred, ignored_label_inds)
    visible_mask = (gt > 0).reshape(-1)

    gt_img = project(xyz, gt, palette, size, axes, depth_axis, visible_mask=visible_mask)
    pred_img = project(xyz, pred, palette, size, axes, depth_axis, visible_mask=visible_mask)
    if gt_img is not None:
        writer.add_image(f"projection_gt",
                            gt_img.transpose(2, 0, 1), epoch)
    if pred_img is not None:
        writer.add_image(f"projection_pred",
                            pred_img.transpose(2, 0, 1), epoch)
