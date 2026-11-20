import json
import os
import stat
import struct
import subprocess
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


@pytest.fixture(autouse=True)
def media_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Photos, videos and the startup sweep stay out of the real Pictures / Videos folders."""
    monkeypatch.setenv("XDG_PICTURES_DIR", str(tmp_path / "Pictures"))
    monkeypatch.setenv("XDG_VIDEOS_DIR", str(tmp_path / "Videos"))


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


def png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """(type, data) of each chunk of a PNG, in file order; checks the signature and every CRC."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, out = 8, list[tuple[bytes, bytes]]()
    while pos < len(data):
        (n,) = struct.unpack_from(">I", data, pos)
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        assert struct.unpack_from(">I", data, pos + 8 + n)[0] == zlib.crc32(kind + body)
        out.append((kind, body))
        pos += 12 + n
    return out


def decode_png(data: bytes) -> Image:
    """8-bit RGB, unfiltered PNG (what png.encode_png writes) back to an H x W x 3 array; checks CRCs."""
    chunks = dict[bytes, bytes]()
    for kind, body in png_chunks(data):
        chunks[kind] = chunks.get(kind, b"") + body
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, ctype) == (8, 2) and b"IEND" in chunks
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(h, 1 + 3 * w)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(h, w, 3)


def fake_v4l2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    fake_exe(tmp_path, monkeypatch, "v4l2-ctl", body)


def fake_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, body: str) -> None:
    """A shell script `name` first on PATH."""
    exe = tmp_path / "bin" / name
    exe.parent.mkdir(exist_ok=True)
    exe.write_text(f"#!/bin/sh\n{body}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")


def ffprobe(path: Path) -> dict[str, object]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)  # type: ignore[no-any-return]
