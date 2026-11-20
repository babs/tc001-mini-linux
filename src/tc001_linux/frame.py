"""Mode-2 frame of the TC001 Mini: 136-byte header + 96x96 little-endian uint16, celsius = raw/64 - 50."""

import struct
from collections.abc import Iterator
from typing import BinaryIO

import numpy as np

from tc001_linux import Plane

SIZE = 96
HEADER = 136
FRAME = HEADER + SIZE * SIZE * 2  # 18568 bytes
MAGIC, MARKER = 0x70827773, 0xAABBCCDD
_MAGIC_BYTES = struct.pack("<I", MAGIC)
_MARKER_BYTES = struct.pack("<I", MARKER)


def parse(frame: bytes) -> Plane | None:
    """The 96x96 Celsius plane, or None if `frame` is not exactly one mode-2 frame."""
    if len(frame) != FRAME or struct.unpack_from("<I", frame, 0)[0] != MAGIC:
        return None
    if struct.unpack_from("<I", frame, HEADER - 4)[0] != MARKER:
        return None
    raw = np.frombuffer(frame, dtype="<u2", offset=HEADER).reshape(SIZE, SIZE)
    return raw.astype(np.float32) / 64.0 - 50.0


def _next_start(buf: bytes) -> int:
    """First magic past offset 0 that can start a frame (marker matches, or not yet readable), or -1."""
    i = buf.find(_MAGIC_BYTES, 1)
    while i >= 0 and i + HEADER <= len(buf) and buf[i + HEADER - 4 : i + HEADER] != _MARKER_BYTES:
        i = buf.find(_MAGIC_BYTES, i + 1)
    return i


def frames(stream: BinaryIO) -> Iterator[bytes]:
    """FRAME-sized chunks of `stream` that start with the magic, resyncing after junk or a short frame."""
    buf = b""
    while chunk := stream.read(FRAME - len(buf)):
        buf += chunk
        if len(buf) < FRAME:
            continue
        # A frame start past offset 0 means a short frame or junk ends there: realign on it.
        # ponytail: a frame short by 1-3 bytes is yielded (>= 410 C last pixel) and drops the next frame;
        # a valid frame with a magic in its last 135 bytes is dropped.
        i = _next_start(buf)
        if i < 0 and buf.startswith(_MAGIC_BYTES) and buf[HEADER - 4 : HEADER] == _MARKER_BYTES:
            yield buf
            buf = b""
        else:
            # No magic: keep a tail that may hold the start of one split across reads.
            buf = buf[i:] if i >= 0 else buf[-(len(_MAGIC_BYTES) - 1) :]
