"""False-colour thermal image: palettes, orientation, upscale with display-only spatial smoothing."""

import functools

import numpy as np
from numpy.typing import NDArray

from tc001_linux import Color, Image, Plane
from tc001_linux.frame import SIZE
from tc001_linux.process import clamp_strength


def lut(stops: list[float], rgb: list[Color]) -> Image:
    """256-entry RGB table, index 0 = coldest, truncated (not rounded) like the Android app."""
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, stops, [c[i] for c in rgb]) for i in range(3)], axis=-1).astype(np.uint8)


PALETTES: dict[str, Image] = {
    "ironbow": lut(
        [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
        [(0, 0, 0), (80, 0, 140), (200, 30, 60), (250, 130, 0), (255, 220, 40), (255, 255, 255)],
    ),
    "white-hot": lut([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]),
    "black-hot": lut([0.0, 1.0], [(255, 255, 255), (0, 0, 0)]),
}


def colorize(norm: Plane, palette: str) -> Image:
    """RGB of `norm` in [0, 1]; np.take on a 256-entry table: per-pixel interp costs 5-7x more."""
    return np.take(PALETTES[palette], (norm * 255).astype(np.uint8), axis=0)


@functools.cache
def _axis(n: int, scale: int) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], Plane]:
    """Nearest index, bilinear (i0, i1, weight) for one axis: pixel-centre aligned, edges clamped."""
    pos = np.clip((np.arange(n * scale) + 0.5) / scale - 0.5, 0, n - 1)
    i0 = np.minimum(pos.astype(np.intp), n - 2)
    return np.arange(n * scale) // scale, i0, i0 + 1, (pos - i0).astype(np.float32)


def upscale(a: Plane, scale: int, strength: float) -> Plane:
    """nearest * (1 - s) + bilinear * s: s = 0 is exactly nearest, s = 1 exactly bilinear."""
    rn, r0, r1, wr = _axis(a.shape[0], scale)
    cn, c0, c1, wc = _axis(a.shape[1], scale)
    near = a[rn][:, cn]
    s = np.float32(clamp_strength(strength))
    if s == 0:
        return near
    one = np.float32(1)
    rows = a[r0] * (one - wr)[:, None] + a[r1] * wr[:, None]
    bilinear = rows[:, c0] * (one - wc) + rows[:, c1] * wc
    out: Plane = near * (one - s) + bilinear * s
    return out


def orient(t: Plane, rotate: int, mirror: bool) -> Plane:
    """Plane turned clockwise by `rotate` degrees, then mirrored left-right."""
    o = np.rot90(t, -(rotate // 90))
    return o[:, ::-1] if mirror else o


def orient_pos(row: int, col: int, rotate: int, mirror: bool) -> tuple[int, int]:
    """Displayed (row, col) of raw pixel (row, col) under `orient`."""
    n = SIZE - 1
    r, c = {0: (row, col), 1: (col, n - row), 2: (n - row, n - col), 3: (n - col, row)}[rotate // 90 % 4]
    return r, n - c if mirror else c


def thermal(
    t: Plane, lo: float, hi: float, palette: str, strength: float, rotate: int, mirror: bool, scale: int
) -> Image:
    """RGB image of `t` in colour range lo..hi (hi > lo), oriented and upscaled by `scale`."""
    norm = np.clip((t - np.float32(lo)) / np.float32(hi - lo), 0, 1)
    return colorize(upscale(orient(norm, rotate, mirror), scale, strength), palette)
