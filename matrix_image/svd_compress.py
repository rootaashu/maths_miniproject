import numpy as np


def _compress_channel(ch, k):
    U, S, Vt = np.linalg.svd(ch, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def svd_compress(img, k):
    if img.ndim == 2:
        return np.clip(_compress_channel(img, k), 0, 1)
    out = np.stack([_compress_channel(img[..., c], k) for c in range(img.shape[2])], axis=-1)
    return np.clip(out, 0, 1)


def singular_values(img):
    gray = img if img.ndim == 2 else img @ np.array([0.299, 0.587, 0.114])
    return np.linalg.svd(gray, compute_uv=False)


def energy_retained(img, k):
    s = singular_values(img)
    return float((s[:k] ** 2).sum() / (s ** 2).sum())


def storage_ratio(shape, k):
    h, w = shape[:2]
    c = shape[2] if len(shape) == 3 else 1
    return (k * (h + w + 1) * c) / (h * w * c)


def psnr(original, approx):
    mse = float(np.mean((original - approx) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
