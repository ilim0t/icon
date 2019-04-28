#!/usr/bin/env python
import argparse
from typing import Tuple

import numpy as np
from utils import show, save, hls_to_rgb, gaussian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, nargs="+", default=(2 ** 13,) * 2,
                        help="the length of one side  of image")
    parser.add_argument("-r", "--radius", type=float, default=0.75, help="the radius of ring")
    parser.add_argument("-c", "--center", type=float, nargs="+", default=(0, 0), help="the center coordinates of ring")
    parser.add_argument("-w", "--line-width", type=float, default=0.05, help="line weight of ring")
    parser.add_argument("-m", "--sigmma", type=float, default=0.01, help="line weight of ring")
    parser.add_argument("-o", "--output", type=str, default="icon.png", help="file name of output")
    args = parser.parse_args()

    img = icon(args.size, 0.6, 0.65, args, np.float32)

    # colorのgridテスト
    # img = np.concatenate([np.concatenate([icon(args.size, l, s, args) for l in np.linspace(0, 1, 21)], 0) for s in
    #                       np.linspace(0, 1, 21)], 1)

    show(img)
    save((img / img.max() * 255).astype(np.uint8), args.output)


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
    del theta

    img[:, :, 3] = ring
    # img[:, :, 3] = 1
    img[:, :, :3] = np.where(0 < ring[:, :, None], color, np.ones_like(color))

    # colorの成分表示
    # img[:, :, :3] = cv2.putText(img[:, :, :3], f"{lightness:.2f}, {saturation:.2f}",
    #                             (int(img.shape[1] * 0.3), int(img.shape[0] * 0.52)),
    #                             cv2.FONT_HERSHEY_SIMPLEX, min(img.shape[:2]) / 400, (0, 0, 0),
    #                             min(img.shape[:2]) // 400, cv2.LINE_AA).get()
    return img


if __name__ == '__main__':
    main()
