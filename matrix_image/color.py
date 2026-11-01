import numpy as np

GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])  # ITU-R BT.601 luma

GRAYSCALE = np.tile(GRAY_WEIGHTS, (3, 1))
SEPIA = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]])
SWAP_RB = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)

COLOR_MATRICES = {"grayscale": GRAYSCALE, "sepia": SEPIA, "swap_rb": SWAP_RB}


def apply_color_matrix(img, M):
    return np.clip(img @ np.asarray(M).T, 0, 1)


def to_grayscale(img):
    return img @ GRAY_WEIGHTS


def negative(img):
    return np.clip(1.0 - img, 0, 1)


def brightness_contrast(img, brightness=0.0, contrast=1.0):
    return np.clip(contrast * (img - 0.5) + 0.5 + brightness, 0, 1)


def apply_mode(img, mode):
    if mode in COLOR_MATRICES:
        return apply_color_matrix(img, COLOR_MATRICES[mode])
    if mode == "negative":
        return negative(img)
    raise ValueError(f"Unknown mode '{mode}'. Choose from {sorted(COLOR_MATRICES) + ['negative']}")
