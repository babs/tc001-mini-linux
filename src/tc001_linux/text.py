"""Labels in DejaVu Sans Mono: glyphs rasterised once by ffmpeg drawtext, then blended with numpy."""

import functools
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tc001_linux import Color, Image, Plane

FONT_FILE = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
CHARS = "0123456789.-°CMinaxetr"  # digits and units, then the window's Min / Max / Center
# DejaVu Sans Mono metrics in font units per em (hhea ascender / descender, advance width).
UNITS, ASCENDER, DESCENDER, ADVANCE = 2048, 1901, 483, 1233
FFMPEG_TIMEOUT = 30.0


class FontError(RuntimeError):
    """Glyphs cannot be rasterised (ffmpeg or the font missing)."""


@dataclass(frozen=True, slots=True)
class Font:
    """Glyph alpha masks (common ink window), pen column and baseline row in a mask, metrics in px."""

    glyphs: dict[str, Plane]
    pen: int
    baseline: int
    advance: int
    ascent: int
    line: int


def _first_line(stderr: bytes) -> str:
    return (stderr.decode(errors="replace").strip().splitlines() or ["?"])[0]


@functools.cache
def font(size: int) -> Font:
    """DejaVu Sans Mono at `size` px; raises FontError with the missing piece named."""
    if not FONT_FILE.is_file():
        raise FontError(f"font not found: {FONT_FILE} (install fonts-dejavu-core)")
    cell, pen, base = 2 * size, size // 2, 3 * size // 2
    vf = ",".join(
        f"drawtext=fontfile={FONT_FILE}:text={ch}:fontsize={size}:fontcolor=white"
        f":x={i * cell + pen}:y={base}-ascent"
        for i, ch in enumerate(CHARS)
    )
    # y_align=baseline needs FFmpeg >= 6.1; y = base - ascent puts the baseline on `base` in 4.4+ too.
    # fmt: off
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=black:s={cell * len(CHARS)}x{cell}",
        "-vf", vf, "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    # fmt: on
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=FFMPEG_TIMEOUT).stdout
    except FileNotFoundError as e:
        raise FontError("ffmpeg not found (install ffmpeg; it rasterises the label glyphs)") from e
    except subprocess.CalledProcessError as e:
        raise FontError(f"ffmpeg drawtext failed: {_first_line(e.stderr)}") from e
    except subprocess.TimeoutExpired as e:
        raise FontError(f"ffmpeg drawtext timed out after {FFMPEG_TIMEOUT:g} s") from e
    if len(raw) != cell * cell * len(CHARS):
        raise FontError(f"ffmpeg drawtext returned {len(raw)} bytes, expected {cell * cell * len(CHARS)}")
    a = np.frombuffer(raw, np.uint8).reshape(cell, len(CHARS), cell).transpose(1, 0, 2).astype(np.float32)
    a /= 255
    rows, cols = np.flatnonzero(a.max(axis=(0, 2))), np.flatnonzero(a.max(axis=(0, 1)))
    if not rows.size:
        raise FontError("ffmpeg drawtext drew no glyphs")
    a = a[:, rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    a.flags.writeable = False
    return Font(
        glyphs=dict(zip(CHARS, a, strict=True)),
        pen=pen - int(cols[0]),
        baseline=base - int(rows[0]),
        advance=round(size * ADVANCE / UNITS),
        ascent=round(size * ASCENDER / UNITS),
        line=round(size * (ASCENDER + DESCENDER) / UNITS),
    )


def text_width(text: str, size: int) -> int:
    return font(size).advance * len(text)


@functools.lru_cache(maxsize=1024)
def _mask(text: str, size: int) -> Plane:
    f = font(size)
    gh, gw = f.glyphs[CHARS[0]].shape
    out = np.zeros((gh, f.advance * (len(text) - 1) + gw), np.float32)
    for i, ch in enumerate(text):
        seg = out[:, i * f.advance : i * f.advance + gw]
        np.maximum(seg, f.glyphs[ch], out=seg)
    out.flags.writeable = False
    return out


def draw_text(img: Image, x: int, baseline: int, text: str, size: int, color: Color) -> None:
    """Blend `text` with its pen at column x and its baseline on row `baseline`; clipped to img."""
    f, m = font(size), _mask(text, size)
    x, y = x - f.pen, baseline - f.baseline
    y0, x0 = max(y, 0), max(x, 0)
    y1, x1 = min(y + m.shape[0], img.shape[0]), min(x + m.shape[1], img.shape[1])
    if y0 >= y1 or x0 >= x1:
        return
    a = m[y0 - y : y1 - y, x0 - x : x1 - x, None]
    reg = img[y0:y1, x0:x1]
    reg[:] = reg + a * (np.array(color, np.float32) - reg) + 0.5
