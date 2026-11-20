"""Command line: the window on the camera or a capture, or `--file CAPTURE --snapshot OUT.png` headless."""

import argparse
import dataclasses
import math
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from tc001_linux import Image
from tc001_linux import settings as cfg
from tc001_linux.capture import DEFAULT_DIRS, media_dir, sweep
from tc001_linux.frame import frames, parse
from tc001_linux.picture import compose
from tc001_linux.png import write_png
from tc001_linux.process import Processor
from tc001_linux.render import PALETTES
from tc001_linux.source import CameraSource, OnFrame, ReplaySource, Source, find_device
from tc001_linux.text import FontError

SNAPSHOT_FRAME = 5  # lets the tracker and the temporal smoothing settle


def snapshot(
    stream: BinaryIO,
    palette: str = "ironbow",
    rotate: int = 0,
    mirror: bool = False,
    smooth: float = cfg.Settings().smoothing,
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
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--device", help="V4L2 node of the camera (default: auto-detect USB 2bdf:0102)")
    src.add_argument("--file", type=Path, help="replay a raw mode-2 capture instead of the camera")
    ap.add_argument("--snapshot", type=Path, metavar="OUT.png", help="write one picture as PNG and exit")
    ap.add_argument("--debug", action="store_true", help="show fps and rejected frames in the window")
    # Window: these override the saved settings, and are saved with the next control change;
    # snapshot: they override the defaults.
    ap.add_argument("--palette", choices=PALETTES)
    ap.add_argument("--rotate", type=int, choices=cfg.ROTATIONS, help="clockwise degrees")
    ap.add_argument(
        "--mirror", action=argparse.BooleanOptionalAction, help="mirror left-right (after rotation)"
    )
    ap.add_argument("--smooth", type=float, metavar="S", help="display smoothing 0-1")
    ap.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"), help="fixed colour range in °C")
    return ap


def _connect(args: argparse.Namespace) -> Callable[[OnFrame], Source | None]:
    """Opens the source when its device or file is there, else None (polled again by the window)."""

    def connect(on_frame: OnFrame) -> Source | None:
        if args.file is not None:
            return ReplaySource(args.file, on_frame) if args.file.is_file() else None
        device = args.device if args.device is not None else find_device()
        return CameraSource(device, on_frame) if device is not None and Path(device).exists() else None

    return connect


def main(argv: list[str] | None = None) -> None:
    ap = _parser()
    args = ap.parse_args(argv)
    if args.smooth is not None and not (math.isfinite(args.smooth) and 0 <= args.smooth <= 1):
        ap.error("--smooth must be between 0 and 1")
    # Raw values span -50..974 C; huge bounds overflow the float32 colour maths into inf / NaN labels.
    if args.range and not all(math.isfinite(v) and abs(v) <= 1e4 for v in args.range):
        ap.error("--range values must be finite, within ±10000")
    if args.range and args.range[1] <= args.range[0]:
        ap.error("--range HI must be above LO")
    try:
        if args.snapshot is not None:
            _snapshot(args, _settings(args, cfg.Settings()))
        else:
            _window(args)
    except (OSError, FontError) as e:
        sys.exit(f"tc001-mini-linux: {e}")


def _settings(args: argparse.Namespace, base: cfg.Settings) -> cfg.Settings:
    """`base` with the options given on the command line."""
    given = (
        ("palette", args.palette),
        ("rotation", args.rotate),
        ("mirror", args.mirror),
        ("smoothing", args.smooth),
    )
    overrides = {k: v for k, v in given if v is not None}
    return dataclasses.replace(base, **overrides)


def _snapshot(args: argparse.Namespace, s: cfg.Settings) -> None:
    if args.file is None:
        sys.exit("tc001-mini-linux: --snapshot renders a capture; add --file CAPTURE")
    with args.file.open("rb") as stream:
        img = snapshot(stream, s.palette, s.rotation, s.mirror, s.smoothing, args.range)
    if img is None:
        sys.exit(f"tc001-mini-linux: {args.file} ends before frame {SNAPSHOT_FRAME}")
    write_png(args.snapshot, img)


def _window(args: argparse.Namespace) -> None:
    if args.file is not None:
        with args.file.open("rb") as stream:
            if next(frames(stream), None) is None:
                sys.exit(f"tc001-mini-linux: no mode-2 frame in {args.file}")
    elif shutil.which("v4l2-ctl") is None:
        sys.exit("tc001-mini-linux: v4l2-ctl not found (install v4l-utils)")
    from tc001_linux.window import run  # Tk only when a window is asked for (001-AC9)

    for kind in DEFAULT_DIRS:
        recovered, discarded = sweep(media_dir(kind))
        for name in recovered:
            print(f"tc001-mini-linux: recovered {name}", file=sys.stderr)
        for name in discarded:
            print(f"tc001-mini-linux: discarded {name}", file=sys.stderr)
    path = cfg.path()
    run(_connect(args), _settings(args, cfg.load(path)), path, args.range, args.debug)
