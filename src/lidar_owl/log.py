# TODO: visu, runtime
import numpy as np
import yaml
from pathlib import Path
import open3d

# TODO: these two functions can be fused and should be attached to the dataset object
def _semkitti_cmap(num_classes: int) -> np.ndarray:
    # gets semantickitti colors from open3d lib 
    resource = Path(open3d._ml3d.__file__).parent / "datasets" / "_resources" / "semantic-kitti.yaml"
    data = yaml.safe_load(resource.read_text())
    # remap colors from preds to original semnantickitti colors
    color_map = {int(k): np.array(v, dtype=np.float32) / 255.0
                for k, v in data["color_map"].items()} 
    inv_map = {int(k): int(v) for k, v in data["learning_map_inv"].items()}
    palette = np.zeros((num_classes, 3), dtype=np.float32)
    for train_id in range(num_classes):
        raw_id = inv_map[train_id]
        palette[train_id] = color_map[raw_id]
    return palette

def _semkitti_train_id_to_name(num_classes: int) -> list[str]:
    """Map train IDs (after learning_map) to human readable SemanticKITTI names."""
    resource = Path(open3d._ml3d.__file__).parent / "datasets" / "_resources" / "semantic-kitti.yaml"
    data = yaml.safe_load(resource.read_text())
    inv_map = {int(k): int(v) for k, v in data["learning_map_inv"].items()}
    labels = {int(k): v for k, v in data["labels"].items()}
    names = []
    for train_id in range(num_classes):
        raw_id = inv_map[train_id]
        names.append(labels[raw_id])
    return names

def _label_names_from_dataset(dataset, num_classes: int) -> list[str]:
    """Prefer dataset-provided label_to_names (train IDs); fallback to SemanticKITTI mapping."""
    names_map = getattr(dataset, "label_to_names", None)
    if isinstance(names_map, dict) and names_map:
        return [names_map.get(i, f"class_{i}") for i in range(num_classes)]
    return _semkitti_train_id_to_name(num_classes)


def _compact_label_names_from_dataset(dataset, num_classes: int, ignored_label_inds) -> list[str]:
    """Names for the compact model label space after ignored labels are removed."""
    full_names = _label_names_from_dataset(dataset, num_classes + len(ignored_label_inds))
    ignored = set(int(label) for label in ignored_label_inds if int(label) >= 0)
    return [name for idx, name in enumerate(full_names) if idx not in ignored][:num_classes]


def _draw(pix, depth_key, pix_labels, width, height, palette):
    # flatten pixel coords for fast per-pixel collision handling
    flat = pix[:, 1] * width + pix[:, 0]

    # sort by pixel id and depth, first point per px is kept
    # For BEV depth_key is -z (highest point wins), for range it is distance (closest point wins)
    order = np.lexsort((depth_key, flat))
    first = np.r_[True, flat[order][1:] != flat[order][:-1]]
    keep = order[first]

    # draw remaining pixels
    canvas = np.zeros((height * width, 3), dtype=np.float32)
    canvas[flat[keep]] = palette[pix_labels[keep]]
    return canvas.reshape(height, width, 3)

def _project_bev(points, labels, size, palette):
    # BEV / Top Down View
    w, h = size
    axes = [0, 1]
    coords = points[:, axes]
    mins = coords.min(0)
    spans = np.maximum(coords.max(0) - mins, 1e-6)
    norm = (coords - mins) / spans
    pix = (norm * np.array([w - 1, h - 1])).round().astype(np.intp, copy=False)
    np.clip(pix, [0, 0], [w - 1, h - 1], out=pix)

    # filter pixels and keep highest one
    depth_key = -points[:, 2]
    return _draw(pix, depth_key, labels, w, h, palette)

def _project_range(points, labels, size, palette):
    # Range / Sensor View

    # adjust image size to LiDAR scan dimensions 
    # width = scan columns (horizontal yaw bins), height = scan rows/rings.
    scan_cols, scan_rows = size

    xy_range = np.hypot(points[:, 0], points[:, 1])
    depth = np.hypot(xy_range, points[:, 2])
    valid = depth > 1e-6
    if not np.any(valid):
        return None

    yaw = np.arctan2(points[valid, 1], points[valid, 0])
    pitch = np.arctan2(points[valid, 2], xy_range[valid])

    # yaw -> columns, pitch -> rows
    # pitch range is taken from the current scan to avoid hardcoding sensor FOV
    pix_x = ((0.5 * (1.0 - yaw / np.pi)) * (scan_cols - 1)).round()
    pitch_min = pitch.min()
    pitch_span = max(float(pitch.max() - pitch_min), 1e-6)
    pix_y = ((1.0 - (pitch - pitch_min) / pitch_span) * (scan_rows - 1)).round()
    pix = np.column_stack((pix_x, pix_y)).astype(np.intp, copy=False)
    np.clip(pix, [0, 0], [scan_cols - 1, scan_rows - 1], out=pix)

    # filter pixels and keep closest one
    return _draw(pix, depth[valid], labels[valid], scan_cols, scan_rows, palette)


def project(points, labels, palette, view="both", bev_size=(512, 512), range_size=(1024, 64), visible_mask=None):
    # Project point labels into an RGB image.
    # size is used for BEV as (cols, rows); range_size should come from the
    # dataset for range view, e.g. SemanticKITTIFlat.range_size = (2048, 64).
    
    if points.size == 0 or labels is None:
        return None
    labels = labels.reshape(-1)

    # keep only valid points (e.g. remove ignored labels, or use provided visible_mask)
    if visible_mask is None:
        mask = labels > 0  # default: 0 = unlabeled/invalid
    else:
        mask = np.asarray(visible_mask, dtype=bool).reshape(-1)

    points = points[mask]
    labels = labels[mask].astype(np.intp, copy=False)
    if points.size == 0:
        return None

    if view == "bev":
        return _project_bev(points, labels, bev_size, palette)
    if view == "range":
        return _project_range(points, labels, range_size, palette)
    if view == "both":
        bev = _project_bev(points, labels, bev_size, palette)
        rng = _project_range(points, labels, range_size, palette)
        if bev is None:
            return rng
        if rng is None:
            return bev
        # BEV and range view usually have different row counts. Pad the smaller
        # image so they can still be logged as one side-by-side TensorBoard image.
        target_h = max(bev.shape[0], rng.shape[0])
        if bev.shape[0] < target_h:
            bev = np.pad(bev, ((0, target_h - bev.shape[0]), (0, 0), (0, 0)))
        if rng.shape[0] < target_h:
            rng = np.pad(rng, ((0, target_h - rng.shape[0]), (0, 0), (0, 0)))
        return np.concatenate((bev, rng), axis=1)

    raise ValueError(f"Unsupported projection view '{view}'. Use 'bev', 'range' or 'both'.")

def log_projection_images(i, points, pred, gt, palette, writer, view="bev", range_size=None):
    # visualizes GT and preds per epoch

    visible_mask = (gt > 0).reshape(-1)

    gt_img = project(points, gt, palette, view=view, visible_mask=visible_mask, range_size=range_size)
    pred_img = project(points, pred, palette, view=view, visible_mask=visible_mask, range_size=range_size)
    if gt_img is not None:
        writer.add_image(f"projection_gt",
                            gt_img.transpose(2, 0, 1), i)
    if pred_img is not None:
        writer.add_image(f"projection_pred",
                            pred_img.transpose(2, 0, 1), i)

