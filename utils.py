#!/usr/bin/env python
from typing import Tuple

from PIL import Image
import numpy as np

try:
    from matplotlib import pyplot
except ImportError:
    def show(*args):
        print("matplotlibがimportできなかったので表示はできません。ただ，プログラムはそれを除き続行されます。")
else:
    def show(*args: Tuple[np.ndarray]):
        for img in args:
            pyplot.imshow(img, interpolation="bilinear")
            pyplot.pause(.01)


def save(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)


def gaussian(x, mean: float, sigmma: float):
    return np.exp(-(x - mean) ** 2 / (2 * sigmma ** 2)) / np.sqrt(2 * np.pi * sigmma ** 2)


def hls_to_rgb(h: np.ndarray, l: np.ndarray, s: np.ndarray):
    assert h.shape == l.shape == s.shape
    m2 = np.where(l <= 0.5, l * (1 + s), l + s - (l * s))
    m1 = 2.0 * l - m2
    return np.stack((_v(m1, m2, h + 1 / 3), _v(m1, m2, h), _v(m1, m2, h - 1 / 3)), 2)


def _v(m1: np.ndarray, m2: np.ndarray, hue: np.ndarray):
    hue = np.mod(hue, 1)
    ret = m1 + (m2 - m1) * hue * 6
    ret = np.where((1 / 6 <= hue) & (hue < 1 / 2), m2, ret)
    ret = np.where((1 / 2 <= hue) & (hue < 2 / 3), m1 + (m2 - m1) * (2 / 3 - hue) * 6, ret)
    ret = np.where(2 / 3 <= hue, m1, ret)
    return ret
