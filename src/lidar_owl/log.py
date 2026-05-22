# TODO: visu, runtime
import numpy as np
from pathlib import Path

def get_git_hash():
    git_dir = Path(__file__).resolve().parents[2] / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head.split(" ", 1)[1]
            ref_path = git_dir / ref
            if ref_path.exists():
                return ref_path.read_text().strip()

            packed = git_dir / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    sha, name = line.split(" ", 1)
                    if name == ref:
                        return sha
            return "unknown"
        return head
    except OSError:
        return "unknown"

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

def _project_bev(points, labels, size, palette, frame_points=None):
    # BEV / Top Down View
    w, h = size
    axes = [0, 1]

    if frame_points is None:
        frame_points = points

    frame_coords = frame_points[:, axes]
    mins = frame_coords.min(0)
    spans = np.maximum(frame_coords.max(0) - mins, 1e-6)

    coords = points[:, axes]
    norm = (coords - mins) / spans
    pix = (norm * np.array([w - 1, h - 1])).round().astype(np.intp, copy=False)
    np.clip(pix, [0, 0], [w - 1, h - 1], out=pix)

    # filter pixels and keep highest one
    depth_key = -points[:, 2]
    return _draw(pix, depth_key, labels, w, h, palette)

def _project_range(points, labels, size, palette, frame_points=None):
    # Range / Sensor View

    # adjust image size to LiDAR scan dimensions 
    # width = scan columns (horizontal yaw bins), height = scan rows/rings.
    scan_cols, scan_rows = size

    if frame_points is None:
        frame_points = points

    frame_xy_range = np.hypot(frame_points[:, 0], frame_points[:, 1])
    frame_depth = np.hypot(frame_xy_range, frame_points[:, 2])
    frame_valid = frame_depth > 1e-6
    if not np.any(frame_valid):
        return None

    xy_range = np.hypot(points[:, 0], points[:, 1])
    depth = np.hypot(xy_range, points[:, 2])
    valid = depth > 1e-6
    if not np.any(valid):
        return None

    yaw = np.arctan2(points[valid, 1], points[valid, 0])
    pitch = np.arctan2(points[valid, 2], xy_range[valid])
    frame_pitch = np.arctan2(frame_points[frame_valid, 2], frame_xy_range[frame_valid])

    # yaw -> columns, pitch -> rows
    # pitch range is taken from the current scan to avoid hardcoding sensor FOV
    pix_x = ((0.5 * (1.0 - yaw / np.pi)) * (scan_cols - 1)).round()
    pitch_min = frame_pitch.min()
    pitch_span = max(float(frame_pitch.max() - pitch_min), 1e-6)
    pix_y = ((1.0 - (pitch - pitch_min) / pitch_span) * (scan_rows - 1)).round()
    pix = np.column_stack((pix_x, pix_y)).astype(np.intp, copy=False)
    np.clip(pix, [0, 0], [scan_cols - 1, scan_rows - 1], out=pix)

    # filter pixels and keep closest one
    return _draw(pix, depth[valid], labels[valid], scan_cols, scan_rows, palette)


def project(points, labels, palette, view="both", bev_size=(512, 512), range_size=(1024, 64), visible_mask=None):
    # Project point labels into an RGB image.
    # size is used for BEV as (cols, rows); range_size should come from the
    # dataset for range view, e.g. SemanticKITTIFlat.range_size = (1024, 64).
    
    if points.size == 0 or labels is None:
        return None
    labels = labels.reshape(-1)

    # Keep the projection frame fixed to the full scan, but draw only valid labels.
    # This keeps GT/prediction images comparable even when their ignore masks differ.
    frame_points = points
    mask = labels > 0  # default: 0 = unlabeled/invalid, so it is not rendered
    if visible_mask is not None:
        mask &= np.asarray(visible_mask, dtype=bool).reshape(-1)

    points = points[mask]
    labels = labels[mask].astype(np.intp, copy=False)

    if points.size == 0:
        return None

    if view == "bev":
        return _project_bev(points, labels, bev_size, palette, frame_points=frame_points)
    if view == "range":
        return _project_range(points, labels, range_size, palette, frame_points=frame_points)
    if view == "both":
        bev = _project_bev(points, labels, bev_size, palette, frame_points=frame_points)
        rng = _project_range(points, labels, range_size, palette, frame_points=frame_points)
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
