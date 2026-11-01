import argparse

from matrix_image import color, filters, svd_compress, transforms
from matrix_image.demo import run_demo
from matrix_image.oi_utils import load_image, make_sample_image, save_image


def _load(path, max_size):
    return make_sample_image(256) if path is None else load_image(path, max_size)


def cmd_transform(a):
    img = _load(a.input, a.max_size)
    steps = []
    if a.flip:
        steps.append(transforms.reflection(a.flip))
    if a.scale != 1.0:
        steps.append(transforms.scaling(a.scale))
    if a.shear_x or a.shear_y:
        steps.append(transforms.shear(a.shear_x, a.shear_y))
    if a.rotate:
        steps.append(transforms.rotation(a.rotate))
    M = transforms.compose(*steps) if steps else transforms.compose()
    print("Combined 3x3 matrix:\n", M.round(4))
    shape = transforms.fit_shape(M, *img.shape[:2])
    out = transforms.affine_transform(img, M, out_shape=shape)
    print("Saved", save_image(out, a.output))


def cmd_filter(a):
    img = _load(a.input, a.max_size)
    save = filters.apply_named(img, a.kernel)
    print("Kernel:\n", filters.KERNELS[a.kernel].round(3))
    print("Saved", save_image(save, a.output))


def cmd_color(a):
    img = _load(a.input, a.max_size)
    out = color.apply_mode(img, a.mode)
    out = color.brightness_contrast(out, a.brightness, a.contrast)
    print("Saved", save_image(out, a.output))


def cmd_svd(a):
    img = _load(a.input, a.max_size)
    out = svd_compress.svd_compress(img, a.k)
    print(f"rank k = {a.k}")
    print(f"storage vs original : {svd_compress.storage_ratio(img.shape, a.k):.1%}")
    print(f"energy retained     : {svd_compress.energy_retained(img, a.k):.2%}")
    print(f"PSNR                : {svd_compress.psnr(img, out):.2f} dB")
    print("Saved", save_image(out, a.output))


def cmd_demo(a):
    img = None if a.input is None else load_image(a.input, a.max_size)
    print("Demo figures written to", run_demo(img, a.output))


def build_parser():
    p = argparse.ArgumentParser(description="Matrix operations & image manipulation")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, out_default):
        sp.add_argument("input", nargs="?", help="input image (default: synthetic sample)")
        sp.add_argument("-o", "--output", default=out_default)
        sp.add_argument("--max-size", type=int, default=512, help="downscale large images")

    s = sub.add_parser("transform", help="rotate / scale / shear / flip")
    common(s, "outputs/transformed.png")
    s.add_argument("--rotate", type=float, default=0.0, help="degrees, counter-clockwise")
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--shear-x", type=float, default=0.0)
    s.add_argument("--shear-y", type=float, default=0.0)
    s.add_argument("--flip", choices=["h", "v"])
    s.set_defaults(func=cmd_transform)

    s = sub.add_parser("filter", help="convolution kernels")
    common(s, "outputs/filtered.png")
    s.add_argument("--kernel", default="gaussian", choices=sorted(filters.KERNELS))
    s.set_defaults(func=cmd_filter)

    s = sub.add_parser("color", help="colour matrices")
    common(s, "outputs/colored.png")
    s.add_argument("--mode", default="sepia", choices=["grayscale", "sepia", "swap_rb", "negative"])
    s.add_argument("--brightness", type=float, default=0.0)
    s.add_argument("--contrast", type=float, default=1.0)
    s.set_defaults(func=cmd_color)

    s = sub.add_parser("svd", help="low-rank compression")
    common(s, "outputs/svd.png")
    s.add_argument("-k", type=int, default=20, help="number of singular values kept")
    s.set_defaults(func=cmd_svd)

    s = sub.add_parser("demo", help="generate gallery figures")
    common(s, "outputs")
    s.set_defaults(func=cmd_demo)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
