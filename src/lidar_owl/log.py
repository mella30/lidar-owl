# TODO: visu, runtime
import numpy as np
import colorsys

def rgb_palette(num_classes: int):
    # creates RGB color palette for PC visu from HSV colorsystem
    palette = np.zeros((num_classes, 3), dtype=np.float32)
    for idx in range(num_classes):
        h = idx / max(1, num_classes)
        palette[idx] = colorsys.hsv_to_rgb(h, 0.75, 1.0)
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