import io
import random
from collections.abc import Callable

import pytest

from tc001_linux.frame import FRAME, HEADER, MAGIC, SIZE, frames, parse


def test_frame0_temperatures_001_AC2(capture: bytes) -> None:
    t = parse(capture[:FRAME])
    assert t is not None and t.shape == (SIZE, SIZE)
    assert float(t.min()) == pytest.approx(22.70, abs=0.01)
    assert float(t.max()) == pytest.approx(23.69, abs=0.01)
    assert float(t.mean()) == pytest.approx(23.04, abs=0.01)


def test_raw_over_64_minus_50_001_AC2(capture: bytes) -> None:
    frame = bytearray(capture[:FRAME])
    frame[HEADER : HEADER + 4] = (0).to_bytes(2, "little") + (64 * 75).to_bytes(2, "little")
    t = parse(bytes(frame))
    assert t is not None and (float(t[0, 0]), float(t[0, 1])) == (-50.0, 25.0)


def test_all_frames_parse_001_AC2(capture: bytes) -> None:
    got = list(frames(io.BytesIO(capture)))
    assert len(got) == len(capture) // FRAME == 5
    assert all(parse(f) is not None for f in got)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda f: b"\0" + f[1:], id="bad-magic"),
        pytest.param(lambda f: f[: HEADER - 4] + b"\0" + f[HEADER - 3 :], id="bad-marker"),
        pytest.param(lambda f: f[:-2], id="short"),
        pytest.param(lambda f: f + b"\0", id="long"),
    ],
)
def test_bad_frames_rejected_001_AC2(capture: bytes, mutate: Callable[[bytes], bytes]) -> None:
    assert parse(mutate(capture[:FRAME])) is None


@pytest.mark.parametrize(
    "junk",
    [
        pytest.param(b"junk", id="leading-junk"),
        pytest.param(b"x" * (FRAME - 1), id="magic-split-1-before-read-end"),
        pytest.param(b"x" * (FRAME - 2), id="magic-split-2-before-read-end"),
        pytest.param(b"x" * (FRAME - 3), id="magic-split-3-before-read-end"),
        pytest.param(b"x" * (3 * FRAME + 7), id="long-junk"),
    ],
)
def test_resync_after_junk_001_AC2(capture: bytes, junk: bytes) -> None:
    got = list(frames(io.BytesIO(junk + capture)))
    assert got == [capture[i : i + FRAME] for i in range(0, len(capture), FRAME)]


@pytest.mark.parametrize(
    ("k", "keep"),
    [
        pytest.param(0, FRAME - 100, id="frame0-short-by-100"),
        pytest.param(2, HEADER + 1000, id="frame2-header-and-some-pixels"),
        pytest.param(1, 8, id="frame1-magic-only"),
        pytest.param(3, FRAME - 4, id="frame3-short-by-4"),
    ],
)
def test_resync_after_short_frame_001_AC2(capture: bytes, k: int, keep: int) -> None:
    """Frame k cut to `keep` bytes: every other frame comes out intact, no garbage frame."""
    whole = [capture[i : i + FRAME] for i in range(0, len(capture), FRAME)]
    data = b"".join(f[:keep] if i == k else f for i, f in enumerate(whole))
    assert list(frames(io.BytesIO(data))) == whole[:k] + whole[k + 1 :]


class Chunked(io.BytesIO):
    """A pipe-like reader: never more than `size` bytes per read."""

    def __init__(self, data: bytes, size: int) -> None:
        super().__init__(data)
        self.size = size

    def read(self, n: int | None = -1) -> bytes:
        return super().read(self.size if n is None or n < 0 else min(n, self.size))


@pytest.mark.parametrize("size", [1000, 997, 3])
def test_short_reads_001_AC2(capture: bytes, size: int) -> None:
    whole = [capture[i : i + FRAME] for i in range(0, len(capture), FRAME)]
    assert list(frames(Chunked(capture, size))) == whole
    assert list(frames(Chunked(b"junk" + capture, size))) == whole


MAGIC_BYTES = MAGIC.to_bytes(4, "little")


@pytest.mark.parametrize(
    "at",
    [
        pytest.param(8, id="header-dup"),
        pytest.param(HEADER + 2000, id="pixel-dup-aligned"),
        pytest.param(HEADER + 5001, id="pixel-dup-odd"),
        pytest.param(FRAME - 136, id="pixel-dup-last-checkable"),
    ],
)
def test_magic_inside_a_valid_frame_keeps_it_001_AC2(capture: bytes, at: int) -> None:
    whole = [capture[i : i + FRAME] for i in range(0, len(capture), FRAME)]
    whole[2] = whole[2][:at] + MAGIC_BYTES + whole[2][at + 4 :]
    assert parse(whole[2]) is not None
    assert list(frames(io.BytesIO(b"".join(whole)))) == whole


def test_magic_in_last_unverifiable_bytes_drops_only_that_frame_001_AC2(capture: bytes) -> None:
    whole = [capture[i : i + FRAME] for i in range(0, len(capture), FRAME)]
    at = FRAME - 135
    whole[2] = whole[2][:at] + MAGIC_BYTES + whole[2][at + 4 :]
    assert list(frames(io.BytesIO(b"".join(whole)))) == whole[:2] + whole[3:]


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(lambda f: MAGIC_BYTES + b"x" * 20000 + f + f, id="magic-led-junk"),
        pytest.param(lambda f: MAGIC_BYTES + b"x" * (FRAME - 5) + f + f, id="magic-junk-split-1"),
        pytest.param(lambda f: MAGIC_BYTES + b"x" * (FRAME - 6) + f + f, id="magic-junk-split-2"),
        pytest.param(lambda f: MAGIC_BYTES + b"x" * (FRAME - 7) + f + f, id="magic-junk-split-3"),
    ],
)
def test_junk_starting_with_a_magic_is_not_a_frame_001_AC2(
    capture: bytes, stream: Callable[[bytes], bytes]
) -> None:
    f = capture[:FRAME]
    assert list(frames(io.BytesIO(stream(f)))) == [f, f]


def tagged(capture: bytes, n: int) -> list[bytes]:
    """n distinct valid frames: capture frames with their first pixel set to their index."""
    base = [capture[i : i + FRAME] for i in range(0, len(capture), FRAME)]
    return [
        (f := base[k % len(base)])[:HEADER] + (k + 1).to_bytes(2, "little") + f[HEADER + 2 :]
        for k in range(n)
    ]


def junk(rng: random.Random) -> bytes:
    """Random bytes holding no magic (0x73 removed), sometimes ending in a partial magic."""
    body = rng.randbytes(rng.choice([1, 3, 135, 136, 2000, FRAME - 1, FRAME + 7])).replace(b"\x73", b"\0")
    return body + MAGIC_BYTES[: rng.randrange(4)]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("size", [7, 1000, 4096, FRAME + 5])
def test_resync_fuzz_001_AC2(capture: bytes, seed: int, size: int) -> None:
    """Junk and frames short by >= 4 bytes between valid ones: all valid frames, in order, nothing else.

    A short frame is always followed by a valid one: short frame + junk is indistinguishable from a frame.
    """
    rng = random.Random(seed)
    valid = tagged(capture, 30)
    parts: list[bytes] = []
    for f in valid:
        roll = rng.random()
        if roll < 0.3:
            parts.append(junk(rng))
        elif roll < 0.6:
            parts.append(rng.choice(valid)[: rng.randrange(4, FRAME - 3)])
        parts.append(f)
    assert list(frames(Chunked(b"".join(parts), size))) == valid
