import os
import stat
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tc001_linux import Image, text

CAPTURE = Path(__file__).parent / "fixtures" / "raw-mode2.bin"


@pytest.fixture(autouse=True)
def headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """001-AC9: every test runs without a display."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


@pytest.fixture
def fresh_font() -> Iterator[None]:
    """Font caches emptied around a test that changes how glyphs are rasterised."""
    text.font.cache_clear()
    text._mask.cache_clear()
    yield
    text.font.cache_clear()
    text._mask.cache_clear()


@pytest.fixture(scope="session")
def capture() -> bytes:
    return CAPTURE.read_bytes()


def decode_png(data: bytes) -> Image:
    """8-bit RGB, unfiltered PNG (what png.encode_png writes) back to an H x W x 3 array; checks CRCs."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, dict[bytes, bytes]()
    while pos < len(data):
        (n,) = struct.unpack_from(">I", data, pos)
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        assert struct.unpack_from(">I", data, pos + 8 + n)[0] == zlib.crc32(kind + body)
        chunks[kind] = chunks.get(kind, b"") + body
        pos += 12 + n
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, ctype) == (8, 2) and b"IEND" in chunks
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(h, 1 + 3 * w)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(h, w, 3)


def fake_v4l2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    exe = tmp_path / "bin" / "v4l2-ctl"
    exe.parent.mkdir(exist_ok=True)
    exe.write_text(f"#!/bin/sh\n{body}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")
