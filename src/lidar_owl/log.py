# TODO: visu, runtime
import numpy as np
import yaml
from pathlib import Path
import open3d

def semkitti_palette(num_classes: int) -> np.ndarray:
    # TODO: the color mapping seems to be off - it works when I remove the "ignore label 0" in the yaml but then, the loss and mIoU is off 

    # gets semantickitti colors from open3d lib 
    resource = Path(open3d._ml3d.__file__).parent / "datasets" / "_resources" / "semantic-kitti.yaml"
    data = yaml.safe_load(resource.read_text())
    # remap colors from preds to original semnantickitti colors
    color_map = {
        int(k): np.array(v[::-1], dtype=np.float32) / 255.0  # BGR -> RGB
        for k, v in data["color_map"].items()
    }
    inv_map = {int(k): int(v) for k, v in data["learning_map_inv"].items()}
    palette = np.zeros((num_classes, 3), dtype=np.float32)
    for train_id in range(num_classes):
        raw_id = inv_map.get(train_id, 0)
        palette[train_id] = color_map.get(raw_id, np.ones(3, dtype=np.float32))
    return palette

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
