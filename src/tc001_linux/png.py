"""Minimal PNG writer (8-bit RGB, no filtering, optional tEXt): zlib + struct, no imaging dependency."""

import struct
import zlib
from pathlib import Path

import numpy as np

from tc001_linux import Image

SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def text_chunk(keyword: str, text: str) -> bytes:
    """A tEXt chunk; ValueError unless the keyword is 1-79 chars, neither holds a NUL and both are Latin-1."""
    if not 1 <= len(keyword) <= 79:
        raise ValueError("tEXt keyword must be 1-79 chars")
    if "\0" in keyword or "\0" in text:
        raise ValueError("tEXt keyword and text must not hold a NUL")
    try:
        data = f"{keyword}\0{text}".encode("latin-1")
    except UnicodeEncodeError:
        raise ValueError("tEXt is Latin-1 only") from None
    return _chunk(b"tEXt", data)


def encode_png(img: Image, text: tuple[str, str] | None = None) -> bytes:
    """PNG bytes of an H x W x 3 uint8 image; `text` = (keyword, text) as a tEXt chunk right after IHDR."""
    h, w, _ = img.shape
    rows = np.zeros((h, 1 + 3 * w), np.uint8)  # leading 0 = filter type None on every row
    rows[:, 1:] = img.reshape(h, 3 * w)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + (text_chunk(*text) if text else b"")
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )


def write_png(path: Path, img: Image) -> None:
    path.write_bytes(encode_png(img))
