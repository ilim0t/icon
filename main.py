#!/usr/bin/env python
import argparse
from typing import Tuple

import numpy as np
from utils import show, save, hls_to_rgb, gaussian


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, nargs="+", default=(2 ** 13,) * 2,
                        help="the length of one side  of image")
    parser.add_argument("-r", "--radius", type=float, default=0.75, help="the radius of ring")
    parser.add_argument("-c", "--center", type=float, nargs="+", default=(0, 0), help="the center coordinates of ring")
    parser.add_argument("-w", "--line-width", type=float, default=0.05, help="line weight of ring")
    parser.add_argument("-m", "--sigmma", type=float, default=1e-4, help="line weight of ring")
    parser.add_argument("-o", "--output", type=str, default="icon.png", help="file name of output")
    args = parser.parse_args()

    img = icon(args.size, 0.6, 0.65, args, np.float32)

    show(img)
    save((img / img.max() * 255).astype(np.uint8), args.output)


def icon_sample(args, num: Tuple[int, int] = (11, 11)) -> np.ndarray:
    img = np.concatenate(
        [np.concatenate([putText(icon(args.size, l, s, args, np.float32), l, s) for l in np.linspace(0, 1, num[0])], 0)
         for s in np.linspace(0, 1, num[1])], 1)
    return img


def icon(size: Tuple[int, int], lightness: float, saturation: float, args, dtype: np.dtype = np.float32) -> np.ndarray:
    assert hasattr(size, "__iter__") and len(size) == 2
    size = tuple(size)
    assert all(isinstance(s, int) and s > 0 for s in size)

    assert 0 <= lightness <= 1
    assert 0 <= saturation <= 1

    # assert isinstance(dtype, np.dtype)

    img = np.zeros(tuple(size) + (4,), dtype=dtype)  # (H, W, C)
    center = args.center
    radius = args.radius
    line_width = args.line_width
    sigmma = args.sigmma

    # bg_color = np.asarray((26, 29, 33), dtype=dtype) / 255
    bg_color = np.asarray((1, 1, 1), dtype=dtype)

    x_coord, y_coord = np.meshgrid(*[np.linspace(-1, 1, s + 2, dtype=dtype)[1:-1] for s in size[::-1]])
    ring = (np.abs(np.sqrt((x_coord - center[0]) ** 2 + (y_coord - center[1]) ** 2) - radius) - line_width <= 0) \
        .astype(dtype)

    outer_distance = np.abs(np.sqrt((x_coord - center[0]) ** 2 + (y_coord - center[1]) ** 2) - radius) - line_width
    ring = np.where(outer_distance > 0, gaussian(outer_distance, 0, sigmma) / gaussian(0, 0, sigmma) + ring, ring)
    del outer_distance

    # 片側gauusian
    # c = np.sqrt((x_coord - center[0]) ** 2 + (y_coord - center[1]) ** 2) - radius
    # d = 0.08
    # ring = np.where(c > 0, gaussian(c, 0, d) / gaussian(0, 0, d), np.zeros_like(ring))

    theta = np.arctan(y_coord / x_coord)
    theta = np.where(x_coord < 0, theta + np.pi, theta)
    del x_coord, y_coord
    theta = np.mod(theta, 2 * np.pi)
    color = hls_to_rgb(theta / (2 * np.pi), np.full_like(theta, lightness), np.full_like(theta, saturation))
    color = np.sqrt(color ** 2 * ring[:, :, None] + bg_color[None, None] ** 2 * (1 - ring[:, :, None]))
    del theta

    img[:, :, 3] = ring
    img[:, :, :3] = color

    return img


def putText(img: np.ndarray, lightness: float, saturation: float) -> np.ndarray:
    import cv2
    ret = img.copy()
    ret[:, :, :3] = cv2.putText(img[:, :, :3], f"{lightness:.2f}, {saturation:.2f}",
                                (int(img.shape[1] * 0.3), int(img.shape[0] * 0.52)),
                                cv2.FONT_HERSHEY_SIMPLEX, min(img.shape[:2]) / 400, (0, 0, 0),
                                min(img.shape[:2]) // 400, cv2.LINE_AA).get()
    ret[:, :, 3] = 1
    return ret


if __name__ == '__main__':
    main()
