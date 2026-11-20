"""Photo name, JSON, tEXt, files and folders; the ffmpeg recorder (001-AC6, 001-AC7)."""

import errno
import fcntl
import json
import os
import struct
import subprocess
import threading
import time
import zlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from tc001_linux import capture
from tc001_linux.capture import (
    Recorder,
    capture_name,
    has_moov,
    media_dir,
    photo_json,
    record_command,
    save_photo,
    sweep,
)
from tc001_linux.picture import HEIGHT, WIDTH
from tc001_linux.png import encode_png, text_chunk
from tc001_linux.process import Readouts, Spot
from tc001_linux.settings import Settings

from .conftest import decode_png, fake_exe, ffprobe, png_chunks

MILLIS = int(datetime(2026, 11, 19, 7, 5, 9, 42_000, UTC).timestamp() * 1000)
PARIS = ZoneInfo("Europe/Paris")


def test_capture_name_001_AC6_AC7() -> None:
    """Same cases as the Android CaptureTest."""
    assert capture_name(MILLIS, UTC) == "TC001_20261119_070509_042"
    assert capture_name(MILLIS, PARIS) == "TC001_20261119_080509_042"
    assert capture_name(MILLIS + 1, UTC) == "TC001_20261119_070509_043"
    assert capture_name(MILLIS + 958, UTC) == "TC001_20261119_070510_000"
    summer = int(datetime(2026, 7, 1, 21, 59, 59, 999_000, UTC).timestamp() * 1000)
    assert capture_name(summer, PARIS) == "TC001_20260701_235959_999"  # CEST, UTC+2
    assert capture_name(summer + 1, PARIS) == "TC001_20260702_000000_000"
    for ms in (MILLIS, summer):  # default: the machine's zone at that instant, DST included
        local = datetime.fromtimestamp(ms / 1000).astimezone()
        assert capture_name(ms) == f"TC001_{local:%Y%m%d_%H%M%S}_{ms % 1000:03d}"


@pytest.mark.parametrize("tz", ["America/New_York", "Asia/Kolkata", "UTC"])
def test_default_zone_follows_tz_001_AC6(monkeypatch: pytest.MonkeyPatch, tz: str) -> None:
    monkeypatch.setenv("TZ", tz)
    time.tzset()
    try:
        assert capture_name(MILLIS) == capture_name(MILLIS, ZoneInfo(tz))
        r, s = readouts(False), Settings()
        assert photo_json(r, s, MILLIS) == photo_json(r, s, MILLIS, ZoneInfo(tz))
    finally:
        monkeypatch.undo()
        time.tzset()


def readouts(locked: bool) -> Readouts:
    return Readouts(Spot(1, 2, 24.25), Spot(3, 4, -0.04), 23.75, 22.7, 23.69, locked)


def test_photo_json_matches_android_format_001_AC6() -> None:
    """Exact text of the Android photoJson: key order, one decimal, no "-0.0", enum palette, offset."""
    s = Settings("white-hot", 270, True, 0.25)
    millis = int(datetime(2026, 11, 20, 7, 5, 9, 42_000, UTC).timestamp() * 1000)
    assert photo_json(readouts(True), s, millis, PARIS) == (
        '{"min":0.0,"center":23.8,"max":24.3,"lo":22.7,"hi":23.7,"palette":"white_hot","rotation":270,'
        '"mirror":true,"smoothing":0.25,"locked":true,"time":"2026-11-20T08:05:09.042+01:00"}'
    )
    d = json.loads(photo_json(readouts(False), Settings(), millis, UTC))
    assert list(d) == [
        "min", "center", "max", "lo", "hi", "palette", "rotation", "mirror", "smoothing", "locked", "time",
    ]  # fmt: skip
    assert d["palette"] == "ironbow" and d["locked"] is False and d["mirror"] is False
    assert d["smoothing"] == 0.5 and d["time"] == "2026-11-20T07:05:09.042Z"
    negative = Readouts(Spot(0, 0, -5.06), Spot(0, 0, -12.34), -7.0, -12.35, -5.0, False)
    text = photo_json(negative, Settings("black-hot", 90, False, 1.0), millis, timezone(timedelta(hours=-5)))
    assert '"min":-12.3,"center":-7.0,"max":-5.1,"lo":-12.4,"hi":-5.0,"palette":"black_hot"' in text
    assert '"smoothing":1.0,' in text and text.endswith('"time":"2026-11-20T02:05:09.042-05:00"}')


def test_text_chunk_after_ihdr_001_AC6() -> None:
    img = np.arange(5 * 3 * 3, dtype=np.uint8).reshape(3, 5, 3)
    data = encode_png(img, ("tc001", '{"min":2.0,"unit":"°C"}'))
    chunks = png_chunks(data)  # CRCs checked
    assert [k for k, _ in chunks] == [b"IHDR", b"tEXt", b"IDAT", b"IEND"]
    assert chunks[1][1] == b'tc001\0{"min":2.0,"unit":"\xb0C"}'  # Latin-1
    assert np.array_equal(decode_png(data), img)
    plain = encode_png(img)
    assert data == plain[:33] + text_chunk("tc001", '{"min":2.0,"unit":"°C"}') + plain[33:]
    assert text_chunk("k", "") == struct.pack(">I", 2) + b"tEXtk\0" + struct.pack(
        ">I", zlib.crc32(b"tEXtk\0")
    )


@pytest.mark.parametrize(
    ("keyword", "text", "error"),
    [
        ("", "x", "1-79"),
        ("k" * 80, "x", "1-79"),
        ("tc\0001", "x", "NUL"),
        ("tc001", "a\0b", "NUL"),
        ("tc001", "20 €", "Latin-1"),
        ("tc€01", "x", "Latin-1"),
    ],
)
def test_text_chunk_rules_001_AC6(keyword: str, text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        text_chunk(keyword, text)


def test_text_chunk_limits_accepted() -> None:
    for keyword in ("k", "k" * 79, "Température"):
        assert text_chunk(keyword, "ÿ").startswith(struct.pack(">I", len(keyword) + 2) + b"tEXt")


def test_save_photo_writes_both_001_AC6(tmp_path: Path) -> None:
    folder = tmp_path / "a" / "TC001"
    assert save_photo(folder, "TC001_x", b"png", b"raw") == folder / "TC001_x.png"
    assert sorted(p.name for p in folder.iterdir()) == ["TC001_x.bin", "TC001_x.png"]
    assert (folder / "TC001_x.png").read_bytes() == b"png" and (folder / "TC001_x.bin").read_bytes() == b"raw"


@pytest.mark.parametrize("failing", [1, 2, 3, 4])
def test_save_photo_both_or_neither_001_AC6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: int
) -> None:
    """A failure right after any write or rename leaves no file, partial or final."""
    calls = 0
    write, replace = capture._write, os.replace

    def counted(f: Callable[..., object]) -> Callable[..., object]:
        def g(*a: object) -> object:
            nonlocal calls
            calls += 1
            out = f(*a)
            if calls == failing:  # after the side effect: the file is there
                raise OSError(28, "No space left on device")
            return out

        return g

    monkeypatch.setattr(capture, "_write", counted(write))
    monkeypatch.setattr(os, "replace", counted(replace))
    with pytest.raises(OSError, match="No space"):
        save_photo(tmp_path, "TC001_x", b"png", b"raw")
    assert list(tmp_path.iterdir()) == []


def test_save_photo_unwritable_folder(tmp_path: Path) -> None:
    (tmp_path / "file").touch()
    with pytest.raises(OSError):
        save_photo(tmp_path / "file" / "TC001", "TC001_x", b"png", b"raw")


def test_media_dir_001_AC6_AC7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = str(tmp_path / "home")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # no xdg-user-dir
    assert media_dir("PICTURES", {"HOME": home, "XDG_PICTURES_DIR": "/data/Img"}) == Path("/data/Img/TC001")
    assert media_dir("PICTURES", {"HOME": home}) == Path(home, "Pictures", "TC001")
    assert media_dir("VIDEOS", {"HOME": home, "XDG_VIDEOS_DIR": "rel"}) == Path(home, "Videos", "TC001")
    fake_exe(tmp_path, monkeypatch, "xdg-user-dir", f'[ "$1" = VIDEOS ] && echo /data/Vid || echo {home}')
    env = {"HOME": home, "XDG_VIDEOS_DIR": "relative/is/ignored"}
    assert media_dir("VIDEOS", env) == Path("/data/Vid/TC001")
    assert media_dir("PICTURES", {"HOME": home}) == Path(home, "Pictures", "TC001")  # $HOME = no entry
    fake_exe(tmp_path, monkeypatch, "xdg-user-dir", "echo relative")
    assert media_dir("VIDEOS", {"HOME": home}) == Path(home, "Videos", "TC001")
    fake_exe(tmp_path, monkeypatch, "xdg-user-dir", "sleep 5")
    monkeypatch.setattr(capture, "XDG_TIMEOUT", 0.1)
    assert media_dir("VIDEOS", {"HOME": home}) == Path(home, "Videos", "TC001")


def test_record_command_001_AC7(tmp_path: Path) -> None:
    cmd = record_command(tmp_path / "out.part")
    joined = " ".join(cmd)
    assert cmd[0] == "ffmpeg" and cmd[-1] == str(tmp_path / "out.part")
    assert "-f rawvideo -pix_fmt rgb24 -s 768x832 -i -" in joined
    assert "-use_wallclock_as_timestamps 1" in joined and "-fps_mode cfr -r 25" in joined
    assert "-an -c:v libx264" in joined and "-pix_fmt yuv420p" in joined and "-f mp4" in joined
    # -r before -i would replace the wall-clock timestamps with a fixed input rate.
    assert joined.index("-r 25") > joined.index("-i -")
    assert "-vf scale=out_color_matrix=bt709:out_range=tv," in joined
    assert "-colorspace bt709 -color_primaries bt709 -color_trc bt709" in joined


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def test_has_moov(tmp_path: Path) -> None:
    f = tmp_path / "v.mp4"
    cases = {
        box(b"ftyp", b"isom") + box(b"mdat", b"x" * 100) + box(b"moov", b"m"): True,
        box(b"ftyp", b"isom")
        + struct.pack(">I", 1)
        + b"mdat"
        + struct.pack(">Q", 24)
        + b"y" * 8
        + box(b"moov", b""): True,
        box(b"ftyp", b"isom") + box(b"mdat", b"x" * 100): False,
        box(b"ftyp", b"isom") + struct.pack(">I", 0) + b"mdat" + b"x" * 20: False,  # runs to the end
        box(b"ftyp", b"isom") + struct.pack(">I", 3) + b"junk" + box(b"moov", b""): False,
        box(b"ftyp", b"isom") + struct.pack(">I", 1) + b"mdat": False,  # truncated large size
        # Index cut short: its header is there, its body is not (crash or full disk while finalising).
        box(b"ftyp", b"isom") + box(b"mdat", b"x" * 100) + box(b"moov", b"m" * 50)[:-10]: False,
        box(b"ftyp", b"isom") + box(b"mdat", b"x" * 100) + box(b"moov", b"m" * 50)[:12]: False,
        # The file stops inside the media data (disk full): the moov after it never made it.
        box(b"ftyp", b"isom") + box(b"mdat", b"x" * 100)[:60]: False,
        box(b"ftyp", b"isom") + struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", 4096) + b"y" * 50: False,
        box(b"ftyp", b"isom") + struct.pack(">I", 1) + b"moov" + struct.pack(">Q", 40) + b"y" * 8: False,
        box(b"ftyp", b"isom") + struct.pack(">I", 1) + b"moov" + struct.pack(">Q", 24) + b"y" * 8: True,
        b"": False,
    }
    for data, want in cases.items():
        f.write_bytes(data)
        assert has_moov(f) is want, data[:40]
    assert not has_moov(tmp_path / "missing.mp4")


def record(folder: Path, seconds: float, fps: float = 25.0) -> tuple[Recorder, list[str], float]:
    """Feeds grey pictures for `seconds`, stops; returns the recorder, its messages and the wall time."""
    done: list[str] = []
    rec = Recorder(folder, "TC001_t", done.append)
    t0 = time.monotonic()
    n = 0
    while time.monotonic() - t0 < seconds:
        rec.feed(bytes([n % 256]) * (WIDTH * HEIGHT * 3))
        n += 1
        time.sleep(1 / fps)
    rec.stop()
    wall = time.monotonic() - t0
    rec.join(capture.FINALISE_TIMEOUT + 5)
    return rec, done, wall


def test_recorder_writes_a_playable_mp4_001_AC7(tmp_path: Path) -> None:
    rec, done, wall = record(tmp_path, 2.0, fps=12)  # half the camera rate: the duration stays real
    assert done == ["Video saved: TC001_t.mp4"] and list(tmp_path.iterdir()) == [rec.path]
    info = ffprobe(rec.path)
    streams = info["streams"]
    assert isinstance(streams, list) and len(streams) == 1  # no audio
    v = streams[0]
    assert (v["codec_name"], v["width"], v["height"], v["pix_fmt"]) == ("h264", 768, 832, "yuv420p")
    assert v["avg_frame_rate"] == "25/1"
    colour = [v.get(k) for k in ("color_range", "color_space", "color_primaries", "color_transfer")]
    assert colour == ["tv", "bt709", "bt709", "bt709"]
    duration = float(info["format"]["duration"])  # type: ignore[index]
    assert abs(duration - wall) < 0.5, (duration, wall)


def test_recorded_colours_decode_back_001_AC7(tmp_path: Path) -> None:
    """Encoded with the BT.709 matrix it is tagged with: decoding by the tags gives the colours back."""
    hot = bytes([255, 77, 109])
    done: list[str] = []
    rec = Recorder(tmp_path, "TC001_c", done.append)
    for _ in range(5):
        rec.feed(hot * (WIDTH * HEIGHT))
        time.sleep(0.04)
    rec.stop()
    rec.join(15)
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(rec.path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    centre = out[3 * (HEIGHT // 2 * WIDTH + WIDTH // 2) :][:3]
    assert max(abs(a - b) for a, b in zip(centre, hot, strict=True)) <= 3, tuple(centre)


def test_empty_recording_deleted_001_AC7(tmp_path: Path) -> None:
    rec, done, _ = record(tmp_path, 0)
    assert done == ["Video failed: no frame recorded"] and list(tmp_path.iterdir()) == []
    rec.feed(b"late")  # after stop: ignored
    assert rec.frames == 0


def test_ffmpeg_error_deletes_the_file_001_AC7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    # Takes two frames and leaves a file without an index, as a crashing encoder would.
    script = 'for a; do last=$a; done; head -c 4000000 > "$last"; echo "Unknown encoder" >&2; exit 1'
    fake_exe(tmp_path, monkeypatch, "ffmpeg", script)
    rec, done, _ = record(out, 0.5)
    assert rec.frames >= 2 and done == ["Video failed: Unknown encoder"] and list(out.iterdir()) == []
    assert rec.stopped


def test_ffmpeg_dying_mid_recording_001_AC7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ffmpeg exits on its own: the writer stops, the recorder reports it, nothing is left."""
    fake_exe(tmp_path, monkeypatch, "ffmpeg", "head -c 100 > /dev/null; exit 3")
    done: list[str] = []
    rec = Recorder(tmp_path / "out", "TC001_t", done.append)
    end = time.monotonic() + 5
    while not rec.stopped:
        assert time.monotonic() < end
        rec.feed(bytes(WIDTH * HEIGHT * 3))
        time.sleep(0.04)
    rec.join(5)
    assert done == ["Video failed: ffmpeg exited with status 3"]
    assert list((tmp_path / "out").iterdir()) == []


def test_stuck_ffmpeg_is_killed_001_AC7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ffmpeg that stops reading: the writer blocks in write(); stop() still ends it within the timeout."""
    fake_exe(tmp_path, monkeypatch, "ffmpeg", 'for a; do last=$a; done; echo x > "$last"; exec sleep 30')
    monkeypatch.setattr(capture, "FINALISE_TIMEOUT", 0.5)
    done: list[str] = []
    rec = Recorder(tmp_path / "out", "TC001_t", done.append)
    for _ in range(capture.QUEUE_FRAMES + 3):
        rec.feed(bytes(WIDTH * HEIGHT * 3))
    assert rec.dropped > 0  # the queue is full: frames are dropped, feed() never waits
    t0 = time.monotonic()
    rec.stop()
    rec.stop()  # idempotent
    rec.join(5)
    assert not rec._thread.is_alive() and time.monotonic() - t0 < 3
    assert done == ["Video failed: ffmpeg did not finish within 0.5 s"]
    assert list((tmp_path / "out").iterdir()) == []


def test_ffmpeg_finalised_by_its_own_sigterm_is_kept_001_AC7(tmp_path: Path) -> None:
    """A service stop signals the whole group: ffmpeg writes the index and exits 255; the file is kept."""
    done: list[str] = []
    rec = Recorder(tmp_path, "TC001_t", done.append)
    for n in range(25):
        rec.feed(bytes([n]) * (WIDTH * HEIGHT * 3))
        time.sleep(0.04)
    rec._proc.terminate()
    rec.stop()  # the viewer got the same SIGTERM
    rec.join(5)
    assert rec._proc.returncode == 255
    assert done == ["Video saved: TC001_t.mp4"], done
    assert ffprobe(rec.path)["streams"][0]["codec_name"] == "h264"  # type: ignore[index]


def test_ffmpeg_ending_while_idle_ends_the_recorder_001_AC7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No frame comes (stalled source) and ffmpeg exits: the writer notices without a write to fail."""
    fake_exe(tmp_path, monkeypatch, "ffmpeg", "echo 'Out of memory' >&2; exit 1")
    done: list[str] = []
    rec = Recorder(tmp_path / "out", "TC001_t", done.append)
    rec.join(5)
    assert rec.stopped and done == ["Video failed: Out of memory"]


def test_missing_ffmpeg_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        Recorder(tmp_path, "TC001_t", print)


def test_rename_failure_reports_and_deletes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(a: object, b: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "replace", refuse)
    _, done, _ = record(tmp_path, 0.3)
    assert done == ["Video failed: [Errno 13] Permission denied"] and list(tmp_path.iterdir()) == []


def test_sweep_recovers_finished_videos_and_drops_parts_001_AC7(tmp_path: Path) -> None:
    """Leftovers of a crash: a part with its whole index is the video; others go; locked ones stay."""
    good = box(b"ftyp", b"isom") + box(b"mdat", b"x" * 10) + box(b"moov", b"m")
    files = {
        ".TC001_a.mp4.part": good,  # finalised, not renamed
        ".TC001_b.mp4.part": good[:-3],  # index cut short
        ".TC001_c.png.part": b"png",  # photo half-written
        ".TC001_c.bin.part": b"raw",
        ".TC001_d.mp4.part": good,  # its final name is taken: never overwritten
        "TC001_d.mp4": b"kept",
        ".TC001_e.mp4.part": good,  # locked: another viewer is recording it
        "notes.part": b"not ours",
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    with (tmp_path / ".TC001_e.mp4.part").open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        recovered, discarded = sweep(tmp_path)
    assert recovered == ["TC001_a.mp4"]
    assert sorted(discarded) == [
        ".TC001_b.mp4.part",
        ".TC001_c.bin.part",
        ".TC001_c.png.part",
        ".TC001_d.mp4.part",
    ]
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == [".TC001_e.mp4.part", "TC001_a.mp4", "TC001_d.mp4", "notes.part"]
    assert (tmp_path / "TC001_a.mp4").read_bytes() == good and (
        tmp_path / "TC001_d.mp4"
    ).read_bytes() == b"kept"
    assert sweep(tmp_path / "missing") == ([], [])


def test_live_recording_survives_a_sweep_001_AC7(tmp_path: Path) -> None:
    """A quiet scene leaves the part untouched for long: only the recorder's lock tells a sweep it is live."""
    done: list[str] = []
    rec = Recorder(tmp_path, "TC001_live", done.append)
    try:
        for n in range(5):
            rec.feed(bytes([n]) * (WIDTH * HEIGHT * 3))
            time.sleep(0.04)
        part = tmp_path / ".TC001_live.mp4.part"
        old = time.time() - 3600
        os.utime(part, (old, old))  # an hour without a write
        assert sweep(tmp_path) == ([], []) and part.exists()
    finally:
        rec.stop()
        rec.join(15)
    assert done == ["Video saved: TC001_live.mp4"] and list(tmp_path.iterdir()) == [rec.path]


def test_lock_failure_leaves_no_part_and_no_fd_001_AC7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """flock unsupported (some NFS / FUSE mounts): the start fails cleanly, nothing is left behind."""

    def no_lock(*_: object) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_lock)
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(OSError):
        Recorder(tmp_path, "TC001_nolock", lambda _: None)
    assert list(tmp_path.iterdir()) == [] and len(os.listdir("/proc/self/fd")) == before


def test_ffmpeg_keeps_the_lock_if_the_viewer_dies_001_AC7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a viewer crash ffmpeg may still be finalising: its inherited lock keeps sweep() away."""
    fake_exe(tmp_path, monkeypatch, "ffmpeg", "cat >/dev/null")
    media = tmp_path / "media"
    media.mkdir()
    rec = Recorder(media, "TC001_orphan", lambda _: None)
    try:
        rec._part_lock.close()  # the viewer's own fd gone, as after SIGKILL
        assert sweep(media) == ([], [])
    finally:
        rec.stop()
        rec.join(15)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_failing_done_is_called_once_and_keeps_the_video_001_AC7(tmp_path: Path) -> None:
    """A `done` that raises (stderr gone) must not turn the saved video into a failure, nor run twice."""
    calls: list[str] = []

    def done(message: str) -> None:
        calls.append(message)
        raise OSError(5, "Input/output error")

    rec = Recorder(tmp_path, "TC001_t", done)
    for n in range(5):
        rec.feed(bytes([n]) * (WIDTH * HEIGHT * 3))
        time.sleep(0.04)
    rec.stop()
    rec.join(15)
    assert calls == ["Video saved: TC001_t.mp4"] and list(tmp_path.iterdir()) == [rec.path]


def test_slow_timer_start_races_the_writer_001_AC7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """stop() from 3 threads while the writer finishes first: one timer start, nothing raised, one outcome."""
    starts: list[int] = []

    class SlowTimer(threading.Timer):
        def __init__(self, *a: object, **k: object) -> None:
            time.sleep(0.3)  # a slow Timer() lets the writer reach cancel() before stop() returns
            super().__init__(*a, **k)  # type: ignore[arg-type]

        def start(self) -> None:
            starts.append(threading.get_ident())
            time.sleep(0.3)
            super().start()

    monkeypatch.setattr(threading, "Timer", SlowTimer)
    done: list[str] = []
    rec = Recorder(tmp_path, "TC001_t", done.append)
    callers = [threading.Thread(target=rec.stop) for _ in range(3)]
    for t in callers:
        t.start()
    for t in callers:
        t.join(5)
    rec.join(15)
    assert len(starts) == 1 and done == ["Video failed: no frame recorded"]
    assert list(tmp_path.iterdir()) == []
