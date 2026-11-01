import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
    "box_blur": np.ones((3, 3)) / 9.0,
    "gaussian": gaussian_kernel(5, 1.2),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "edge": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
}


def convolve2d(channel, kernel):
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")
    ph, pw = kh // 2, kw // 2
    padded = np.pad(channel, ((ph, ph), (pw, pw)), mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel[::-1, ::-1])


def apply_kernel(img, kernel):
    if img.ndim == 2:
        return convolve2d(img, kernel)
    return np.stack([convolve2d(img[..., c], kernel) for c in range(img.shape[2])], axis=-1)


def apply_named(img, name):
    if name not in KERNELS:
        raise ValueError(f"Unknown kernel '{name}'. Choose from {sorted(KERNELS)}")
    out = apply_kernel(img, KERNELS[name])
    if name in ("emboss", "sobel_x", "sobel_y"):
        out = out + 0.5  # re-centre signed responses so they are visible
    return np.clip(out, 0, 1)


def sobel_edges (gray):
    gx = convolve2d(gray, KERNELS["sobel_x"])
    gy = convolve2d(gray, KERNELS["sobel_y"])
    mag = np.hypot(gx, gy)
    return mag / mag.max() if mag.max() > 0 else mag
