"""Command line: `tc001-mini-linux --file CAPTURE --snapshot OUT.png` renders one picture headless."""

import argparse
import math
import sys
from pathlib import Path
from typing import BinaryIO

from tc001_linux import Image
from tc001_linux.frame import frames, parse
from tc001_linux.picture import compose
from tc001_linux.png import write_png
from tc001_linux.process import Processor
from tc001_linux.render import PALETTES
from tc001_linux.text import FontError

SNAPSHOT_FRAME = 5  # lets the tracker and the temporal smoothing settle


def snapshot(
    stream: BinaryIO,
    palette: str = "ironbow",
    rotate: int = 0,
    mirror: bool = False,
    smooth: float = 0.0,
    fixed: tuple[float, float] | None = None,
) -> Image | None:
    """Picture of the SNAPSHOT_FRAME-th parsed frame of `stream`, or None if the stream ends first."""
    proc = Processor(fixed)
    parsed = (t for t in map(parse, frames(stream)) if t is not None)
    for n, t in enumerate(parsed, 1):
        shown, r = proc.process(t, smooth)
        if n == SNAPSHOT_FRAME:
            return compose(shown, r, palette, smooth, rotate, mirror)
    return None


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tc001-mini-linux", description="TOPDON TC001 Mini thermal viewer.")
    ap.add_argument("--file", type=Path, help="replay a raw mode-2 capture instead of the camera")
    ap.add_argument("--snapshot", type=Path, metavar="OUT.png", help="write one picture as PNG and exit")
    ap.add_argument("--palette", choices=PALETTES, default="ironbow")
    ap.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0, help="clockwise degrees")
    ap.add_argument("--mirror", action="store_true", help="mirror left-right (after rotation)")
    ap.add_argument("--smooth", type=float, default=0.0, metavar="S", help="display smoothing 0-1")
    ap.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"), help="fixed colour range in °C")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = _parser()
    args = ap.parse_args(argv)
    if not (math.isfinite(args.smooth) and 0 <= args.smooth <= 1):
        ap.error("--smooth must be between 0 and 1")
    # Raw values span -50..974 C; huge bounds overflow the float32 colour maths into inf / NaN labels.
    if args.range and not all(math.isfinite(v) and abs(v) <= 1e4 for v in args.range):
        ap.error("--range values must be finite, within ±10000")
    if args.range and args.range[1] <= args.range[0]:
        ap.error("--range HI must be above LO")
    if args.snapshot is None:
        sys.exit("tc001-mini-linux: the window is not implemented yet; use --file CAPTURE --snapshot OUT.png")
    if args.file is None:
        sys.exit("tc001-mini-linux: the camera source is not implemented yet; use --file CAPTURE")
    try:
        with args.file.open("rb") as stream:
            img = snapshot(stream, args.palette, args.rotate, args.mirror, args.smooth, args.range)
        if img is None:
            sys.exit(f"tc001-mini-linux: {args.file} ends before frame {SNAPSHOT_FRAME}")
        write_png(args.snapshot, img)
    except (OSError, FontError) as e:
        sys.exit(f"tc001-mini-linux: {e}")
