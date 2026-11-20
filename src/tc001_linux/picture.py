"""The 768x832 picture of the Android app: thermal image with crosshair and markers over the scale band."""

import functools
import math

import numpy as np

from tc001_linux import Color, Image, Plane
from tc001_linux.frame import SIZE
from tc001_linux.process import Readouts
from tc001_linux.render import PALETTES, orient_pos, thermal
from tc001_linux.text import draw_text, font, text_width

SCALE = 8
SIDE = SIZE * SCALE
BAND = 8 * SCALE
WIDTH, HEIGHT = SIDE, SIDE + BAND  # multiples of 16 for the H.264 encoder
BAR_MARGIN = 2 * SCALE
BAR_WIDTH = SIDE - 2 * BAR_MARGIN
BAR_TOP, BAR_HEIGHT = SIDE + SCALE, 3 * SCALE
FONT_SIZE = round(2.5 * SCALE)
HALF = 3 * SCALE  # marker half-side, crosshair half-arm
# Even stroke widths keep each line centred on the image centre / spot centre pixel boundary.
STROKE = max(2, 2 * round(SCALE / 3))
OUTLINE = max(1, round(SCALE / 6))

HOT: Color = (0xFF, 0x4D, 0x6D)
COLD: Color = (0x00, 0xC8, 0xFF)
CENTRE: Color = (0xFF, 0xFF, 0xFF)
PANEL = 24
UNIT = "°C"
LONG_MAX = 2**63 - 1


def format_celsius(c: float) -> str:
    """One decimal, half away from zero and no "-0.0", like the Android app's formatCelsius.

    Java's Math.round: NaN gives 0, out-of-range values saturate at Long.MAX_VALUE.
    """
    x = abs(c) * 10
    tenths = 0 if math.isnan(x) else LONG_MAX if x >= 2.0**63 else math.floor(x + 0.5)
    return f"{'-' if c < 0 and tenths else ''}{tenths // 10}.{tenths % 10}"


def stroke(img: Image, rects: list[tuple[int, int, int, int]], color: Color) -> None:
    """Paint the union of rects (y0, y1, x0, x1) in `color` over a dark outline OUTLINE px wide all round.

    One mask for the union: darkening each rect's outline separately would darken twice where they cross.
    """
    h, w = img.shape[:2]
    wy0, wx0 = max(min(r[0] for r in rects) - OUTLINE, 0), max(min(r[2] for r in rects) - OUTLINE, 0)
    wy1, wx1 = min(max(r[1] for r in rects) + OUTLINE, h), min(max(r[3] for r in rects) + OUTLINE, w)
    if wy0 >= wy1 or wx0 >= wx1:
        return
    win = img[wy0:wy1, wx0:wx1]
    halo, core = np.zeros(win.shape[:2], bool), np.zeros(win.shape[:2], bool)

    def put(m: np.ndarray[tuple[int, int], np.dtype[np.bool]], y0: int, y1: int, x0: int, x1: int) -> None:
        m[max(y0 - wy0, 0) : max(y1 - wy0, 0), max(x0 - wx0, 0) : max(x1 - wx0, 0)] = True

    for y0, y1, x0, x1 in rects:
        put(halo, y0 - OUTLINE, y1 + OUTLINE, x0 - OUTLINE, x1 + OUTLINE)
        put(core, y0, y1, x0, x1)
    win[halo & ~core] //= 3  # ~ Android's 0xB0 black
    win[core] = color


def badge_width(text: str) -> int:
    return text_width(text, FONT_SIZE) + SCALE


def badge(img: Image, x: int, y: int, text: str, color: Color) -> None:
    """`text` on a dark backdrop with top-left (x, y), clamped inside img so it stays readable."""
    f = font(FONT_SIZE)
    w = badge_width(text)
    x, y = min(max(x, 0), img.shape[1] - w), min(max(y, 0), img.shape[0] - f.line)
    img[y : y + f.line, x : x + w] //= 4  # ~ Android's 0xC0 black
    draw_text(img, x + SCALE // 2, y + f.ascent, text, FONT_SIZE, color)


def marker_box(y: int, x: int) -> list[tuple[int, int, int, int]]:
    """Four sides of the square marker centred on display pixel centre (y, x)."""
    h2 = STROKE // 2
    a0, a1, b0, b1 = y - HALF - h2, y + HALF + h2, x - HALF - h2, x + HALF + h2
    return [
        (a0, a0 + STROKE, b0, b1),
        (a1 - STROKE, a1, b0, b1),
        (a0, a1, b0, b0 + STROKE),
        (a0, a1, b1 - STROKE, b1),
    ]


def marker_label_x(x: int, text: str) -> int:
    """Badge left edge: right of the marker, or left of it when it would leave the image."""
    gap = HALF + STROKE // 2 + OUTLINE + 2  # clear of the outline, not just the stroke
    w = badge_width(text)
    return x + gap if x + gap + w <= SIDE else x - gap - w


@functools.cache
def bar_row(palette: str) -> Image:
    """Scale-bar row, cold (LUT index 0) left, hot right; float32 index maths as the Android app."""
    i = np.arange(BAR_WIDTH, dtype=np.float32) / np.float32(BAR_WIDTH - 1) * np.float32(255)
    row = PALETTES[palette][i.astype(np.intp)]
    row.flags.writeable = False
    return row


def scale_labels(lo: float, mid: float, hi: float) -> list[tuple[str, int]]:
    """(text, pen x) of the lo / mid / hi°C labels: left-, centre- and right-aligned on the bar."""
    lo_s, mid_s, hi_s = format_celsius(lo), format_celsius(mid), format_celsius(hi) + UNIT
    right = BAR_MARGIN + BAR_WIDTH
    return [
        (lo_s, BAR_MARGIN),
        (mid_s, SIDE // 2 - text_width(mid_s, FONT_SIZE) // 2),
        (hi_s, right - text_width(hi_s, FONT_SIZE)),
    ]


def compose(shown: Plane, r: Readouts, palette: str, strength: float, rotate: int, mirror: bool) -> Image:
    """The HEIGHT x WIDTH RGB picture of plane `shown` with the markers and numbers of `r`."""
    out = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    img = out[:SIDE]
    img[:] = thermal(shown, r.lo, r.hi, palette, strength, rotate, mirror, SCALE)

    c, h2 = SIDE // 2, STROKE // 2
    stroke(img, [(c - h2, c + h2, c - HALF, c + HALF), (c - HALF, c + HALF, c - h2, c + h2)], CENTRE)
    # Lower-right of the crosshair, clear of both arms; before the markers so they stay on top.
    badge(img, c + SCALE + 2, c + SCALE + 2, format_celsius(r.centre), CENTRE)
    for spot, color in ((r.hot, HOT), (r.cold, COLD)):
        row, col = orient_pos(spot.row, spot.col, rotate, mirror)
        y, x = row * SCALE + SCALE // 2, col * SCALE + SCALE // 2
        stroke(img, marker_box(y, x), color)
        text = format_celsius(spot.celsius)
        badge(img, marker_label_x(x, text), y - HALF, text, color)

    out[SIDE:] = PANEL
    out[BAR_TOP : BAR_TOP + BAR_HEIGHT, BAR_MARGIN : BAR_MARGIN + BAR_WIDTH] = bar_row(palette)
    baseline = BAR_TOP + BAR_HEIGHT + font(FONT_SIZE).ascent + SCALE // 4
    for text, x in scale_labels(r.lo, r.mid, r.hi):
        draw_text(out, x, baseline, text, FONT_SIZE, CENTRE)
    return out
