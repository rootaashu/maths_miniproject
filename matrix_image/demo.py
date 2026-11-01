from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import color, filters, svd_compress, transforms
from .io_utils import make_sample_image, save_image


def _gallery(items, title, path, ncols=4):
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (label, im) in zip(axes, items):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_demo(img=None, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = make_sample_image(256) if img is None else img
    save_image(img, out / "original.png")

    combo = transforms.compose(
        transforms.rotation(20), transforms.scaling(0.8), transforms.shear(0.2, 0.0)
    )
    _gallery(
        [
            ("original", img),
            ("rotate 30°", transforms.rotate(img, 30)),
            ("scale 0.6x", transforms.scale(img, 0.6)),
            ("shear x=0.4", transforms.shear_image(img, shx=0.4)),
            ("flip horizontal", transforms.flip(img, "h")),
            ("flip vertical", transforms.flip(img, "v")),
            ("translate (40, 20)", transforms.translate(img, 40, 20)),
            ("rotate+scale+shear", transforms.affine_transform(img, combo)),
        ],
        "Geometric transforms (3x3 homogeneous matrices)",
        out / "transforms.png",
    )

    gray = color.to_grayscale(img)
    _gallery(
        [
            ("original", img),
            ("gaussian blur", filters.apply_named(img, "gaussian")),
            ("sharpen", filters.apply_named(img, "sharpen")),
            ("edge detect", filters.apply_named(img, "edge")),
            ("emboss", filters.apply_named(img, "emboss")),
            ("sobel x", filters.apply_named(gray, "sobel_x")),
            ("sobel y", filters.apply_named(gray, "sobel_y")),
            ("sobel magnitude", filters.sobel_edges(gray)),
        ],
        "Convolution filters (kernel matrices)",
        out / "filters.png",
    )

     _gallery(
        [
            ("original", img),
            ("grayscale", color.apply_mode(img, "grayscale")),
            ("sepia", color.apply_mode(img, "sepia")),
            ("swap R<->B", color.apply_mode(img, "swap_rb")),
            ("negative", color.apply_mode(img, "negative")),
            ("brighter", color.brightness_contrast(img, brightness=0.2)),
            ("high contrast", color.brightness_contrast(img, contrast=1.6)),
        ],
        "Colour operations (3x3 pixel matrices)",
        out / "color.png",
    )

    ks = [2, 5, 10, 25, 50]
    items = [("original", img)]
    for k in ks:
        approx = svd_compress.svd_compress(img, k)
        ratio = svd_compress.storage_ratio(img.shape, k)
        items.append((f"k={k}  size={ratio:.0%}\nPSNR={svd_compress.psnr(img, approx):.1f} dB", approx))
    _gallery(items, "SVD low-rank approximation", out / "svd.png", ncols=3)

    s = svd_compress.singular_values(img)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogy(s)
    ax[0].set(title="Singular values (log scale)", xlabel="index", ylabel="σ")
    ax[1].plot(np.cumsum(s ** 2) / np.sum(s ** 2))
    ax[1].set(title="Cumulative energy retained", xlabel="k", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(out / "svd_spectrum.png", dpi=110)
    plt.close(fig)
    return out
