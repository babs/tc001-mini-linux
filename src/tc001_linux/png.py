"""Minimal PNG writer (8-bit RGB, no filtering): zlib + struct, no imaging dependency."""

import struct
import zlib
from pathlib import Path

import numpy as np

from tc001_linux import Image


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_png(img: Image) -> bytes:
    """PNG bytes of an H x W x 3 uint8 image."""
    h, w, _ = img.shape
    rows = np.zeros((h, 1 + 3 * w), np.uint8)  # leading 0 = filter type None on every row
    rows[:, 1:] = img.reshape(h, 3 * w)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )


def write_png(path: Path, img: Image) -> None:
    path.write_bytes(encode_png(img))
