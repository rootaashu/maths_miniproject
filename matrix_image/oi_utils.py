

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path, max_size=None):
    """Load an image as an (H, W, 3) float array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    if max_size:
        img.thumbnail((max_size, max_size))
    return np.asarray(img, dtype=np.float64) / 255.0


def save_image(arr, path):
    """Save a float array in [0, 1] (H, W) or (H, W, 3) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out).save(path)
    return path


def make_sample_image(size=256):
    """Synthetic test image (asymmetric on purpose so rotations/flips are obvious)."""
    y, x = np.mgrid[0:size, 0:size] / (size - 1.0)
    img = np.stack([x, y, 1.0 - 0.5 * (x + y)], axis=-1)

    def rect(x0, y0, x1, y1, color):
        a, b, c, d = (int(v * size) for v in (x0, y0, x1, y1))
        img[b:d, a:c] = color

    rect(0.15, 0.15, 0.25, 0.85, (1, 1, 1))   # letter "F": stem
    rect(0.15, 0.15, 0.60, 0.25, (1, 1, 1))   # top bar
    rect(0.15, 0.45, 0.50, 0.55, (1, 1, 1))   # middle bar
    yy, xx = np.mgrid[0:size, 0:size]
    circle = (xx - 0.72 * size) ** 2 + (yy - 0.72 * size) ** 2 < (0.13 * size) ** 2
    img[circle] = (0.95, 0.8, 0.1)
    img[::32, :] = 0.0                        # grid lines
    img[:, ::32] = 0.0
    return img
