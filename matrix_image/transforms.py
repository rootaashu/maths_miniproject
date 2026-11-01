import numpy as np

from .matrix_ops import inverse, mutmul


def translation(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def rotation(degrees):
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def scaling(sx, sy=None):
    sy = sx if sy is None else sy
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def shear(shx=0.0, shy=0.0):
    return np.array([[1.0, shx, 0.0], [shy, 1.0, 0.0], [0.0, 0.0, 1.0]])


def reflection(axis):

    if axis == "h":
        return scaling(-1.0, 1.0)
    if axis == "v":
        return scaling(1.0, -1.0)
    raise ValueError("axis must be 'h' or 'v'")


def compose(*matrices):
    result = np.eye(3)
    for M in matrices:
        result = mutmul(M, result)
    return result

def _fetch(img, xi, yi, fill):
    h, w, c = img.shape
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full((xi.size, c), fill, dtype=np.float64)
    out[valid] = img[yi[valid], xi[valid]]
    return out


def _sample(img, sx, sy, fill, interpolation):
    if interpolation == "nearest":
        return _fetch(img, np.rint(sx).astype(int), np.rint(sy).astype(int), fill)
    if interpolation != "bilinear":
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    wx, wy = (sx - x0)[:, None], (sy - y0)[:, None]
    return (
        _fetch(img, x0, y0, fill) * (1 - wx) * (1 - wy)
        + _fetch(img, x0 + 1, y0, fill) * wx * (1 - wy)
        + _fetch(img, x0, y0 + 1, fill) * (1 - wx) * wy
        + _fetch(img, x0 + 1, y0 + 1, fill) * wx * wy
    )

def affine_transform(img, M, out_shape=None, about_center=True, fill=0.0,
                     interpolation="bilinear"):
    gray = img.ndim == 2
    src = img[..., None] if gray else img
    h, w, c = src.shape
    oh, ow = out_shape if out_shape else (h, w)

    if about_center:
        M = compose(
            translation(-(w - 1) / 2, -(h - 1) / 2),
            M,
            translation((ow - 1) / 2, (oh - 1) / 2),
        )

    Minv = inverse(M)
    xs, ys = np.meshgrid(np.arange(ow), np.arange(oh))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])  # 3 x N
    mapped = Minv @ pts  # one big product: NumPy for speed
    sx, sy = mapped[0] / mapped[2], mapped[1] / mapped[2]

    out = _sample(src, sx, sy, fill, interpolation).reshape(oh, ow, c)
    return out[..., 0] if gray else out

def rotate(img, degrees, expand=True, **kw):
    h, w = img.shape[:2]
    shape = None
    if expand:
        t = np.deg2rad(degrees)
        c, s = abs(np.cos(t)), abs(np.sin(t))
        shape = (int(np.ceil(w * s + h * c)), int(np.ceil(w * c + h * s)))
    return affine_transform(img, rotation(degrees), out_shape=shape, **kw)


def scale(img, sx, sy=None, **kw):
    sy = sx if sy is None else sy
    h, w = img.shape[:2]
    shape = (max(1, round(h * sy)), max(1, round(w * sx)))
    return affine_transform(img, scaling(sx, sy), out_shape=shape, **kw)


def shear_image(img, shx=0.0, shy=0.0, **kw):
    h, w = img.shape[:2]
    shape = (int(np.ceil(h + abs(shy) * w)), int(np.ceil(w + abs(shx) * h)))
    return affine_transform(img, shear(shx, shy), out_shape=shape, **kw)


def flip(img, axis="h", **kw):
    return affine_transform(img, reflection(axis), **kw)


def translate(img, tx, ty, **kw):
    return affine_transform(img, translation(tx, ty), about_center=False, **kw)


def fit_shape(M, h , w):
    corners = np.array([[0, w-1, w - 1, 0],
                        [0, 0, h - 1, h -1],
                        [1, 1, 1, 1]], dtype=float)
    corners[0] -= (w-1) / 2
    corners[1] -= (h - 1) /2 
    p = M @ corners
    return (int(np.ceil(np.ptp(p[1]))) + 1, int(np.ceil(np.ptp(p[0]))) +1)

