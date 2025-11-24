# TODO: visu, runtime
import numpy as np
import yaml
from pathlib import Path
import open3d

import util

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
        raw_id = inv_map.get(train_id, 0)
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
        raw_id = inv_map.get(train_id, -1)
        names.append(labels.get(raw_id, f"raw_{raw_id}"))
    return names

def label_names_from_dataset(dataset, num_classes: int) -> list[str]:
    """Prefer dataset-provided label_to_names (train IDs); fallback to SemanticKITTI mapping."""
    names_map = getattr(dataset, "label_to_names", None)
    if isinstance(names_map, dict) and names_map:
        return [names_map.get(i, f"class_{i}") for i in range(num_classes)]
    return semkitti_train_id_to_name(num_classes)

def project(points, labels, palette, size=(512, 512), axes=(0, 1), depth_axis=2):
    # BEV projection -> TODO: sensor view projection?
    if points.size == 0 or labels is None:
        return None
    mask = (labels >= 0).squeeze()
    pts = points[mask, :]
    lbs = labels[mask, :]
    if pts.size == 0:
        return None
    w, h = size
    coords = pts[:, axes]
    mins = coords.min(0)
    spans = np.maximum(coords.max(0) - mins, 1e-6)
    norm = (coords - mins) / spans
    pix = np.clip((norm * np.array([w - 1, h - 1])).round().astype(int),
                  [0, 0], [w - 1, h - 1])
    depth = pts[:, depth_axis] if depth_axis is not None else np.arange(len(pts))
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    zbuf = np.full((h, w), -np.inf)
    for (x, y), z, label in zip(pix, depth, lbs):
        if z >= zbuf[y, x]:
            zbuf[y, x] = z
            canvas[y, x] = palette[label]
    return canvas

def log_projection_images(epoch, summary, cfg, palette, writer):
    # TODO: clean that mess up..
    # visualizes GT and preds per epoch
    if not cfg.get('enabled', True):
        return
    stages = cfg.get('record_for', list(summary.keys()))
    size = tuple(cfg.get('image_size', [512, 512]))
    axes = tuple(cfg.get('axes', [0, 1])) 
    depth_axis = cfg.get('depth_axis', 2)

    # TODO: heuristic for which PCs to log (not random)
    for stage in stages:
        stage_summary = summary.get(stage, {})
        sem = stage_summary.get('semantic_segmentation')
        if not sem:
            continue
    
        # take only first sample from batched data
        xyz = util.tensor_to_np(sem.get('vertex_positions'))[0, :, :]
        gt = util.tensor_to_np(sem.get('vertex_gt_labels'))[0, :, :]
        pred = util.tensor_to_np(sem.get('vertex_predict_labels'))[0, :, :]
    
        gt_img = project(xyz, gt, palette, size, axes, depth_axis) 
        pred_img = project(xyz, pred, palette, size, axes, depth_axis)
        if gt_img is not None:
            writer.add_image(f"{stage}/projection_gt",
                                gt_img.transpose(2, 0, 1), epoch)
        if pred_img is not None:
            writer.add_image(f"{stage}/projection_pred",
                                pred_img.transpose(2, 0, 1), epoch)
