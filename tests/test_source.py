import os
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tc001_linux import Plane, source
from tc001_linux.frame import FRAME
from tc001_linux.source import (
    CameraSource,
    Mailbox,
    ReplaySource,
    Source,
    Step,
    Watch,
    camera_command,
    find_device,
    last_message,
)

from .conftest import CAPTURE, fake_v4l2


class Sink:
    def __init__(self) -> None:
        self.raws: list[bytes] = []
        self.got = threading.Event()

    def __call__(self, raw: bytes, t: Plane) -> None:
        assert t.shape == (96, 96)
        self.raws.append(raw)
        self.got.set()


def wait_for(cond: object, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not cond():  # type: ignore[operator]
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_mailbox_keeps_latest_001_AC1() -> None:
    box = Mailbox[int]()
    assert box.take() is None
    box.put(1)
    box.put(2)
    box.put(3)
    assert box.take() == 3 and box.take() is None and box.dropped == 2
    box.put(4)
    assert box.take() == 4 and box.dropped == 2


def test_mailbox_never_blocks_the_writer() -> None:
    box = Mailbox[int]()

    def write() -> None:
        for i in range(1000):
            box.put(i)

    writer = threading.Thread(target=write)
    writer.start()
    writer.join(5)  # nobody takes: the writer still finishes
    assert not writer.is_alive() and box.take() == 999 and box.dropped == 999


def fake_sysfs(root: Path, nodes: dict[str, tuple[str, str, str]]) -> Path:
    """video4linux tree: node -> (vendor, product, index); `device` links to a USB interface dir."""
    sysfs = root / "class" / "video4linux"
    sysfs.mkdir(parents=True)
    for i, (name, (vendor, product, index)) in enumerate(nodes.items()):
        usb = root / "devices" / f"usb{i}"
        iface = usb / f"1-{i}:1.0"
        iface.mkdir(parents=True)
        (usb / "idVendor").write_text(vendor + "\n")
        (usb / "idProduct").write_text(product + "\n")
        node = sysfs / name
        node.mkdir()
        (node / "index").write_text(index + "\n")
        (node / "device").symlink_to(iface)
    return sysfs


def test_find_device_001_AC1(tmp_path: Path) -> None:
    sysfs = fake_sysfs(
        tmp_path,
        {
            "video0": ("046d", "0944", "0"),  # another webcam
            "video1": ("046d", "0944", "1"),
            "video2": ("2bdf", "0102", "1"),  # the TC001's metadata node
            "video3": ("2bdf", "0102", "0"),
        },
    )
    assert find_device(sysfs) == "/dev/video3"
    (sysfs / "video3" / "index").unlink()  # half-gone node during an unplug
    assert find_device(sysfs) is None
    assert find_device(tmp_path / "nothing") is None


def test_camera_command_001_AC1() -> None:
    assert camera_command("/dev/video7") == [
        "v4l2-ctl",
        "-d",
        "/dev/video7",
        "--set-fmt-video=width=4,height=2321,pixelformat=YUYV",
        "--stream-mmap",
        "--stream-to=-",
    ]


def test_replay_loops_and_stops_001_AC9(capture: bytes) -> None:
    sink = Sink()
    t0 = time.monotonic()
    src = ReplaySource(CAPTURE, sink)
    wait_for(lambda: len(sink.raws) >= 12)  # more than the 5 frames of the capture: it loops
    elapsed = time.monotonic() - t0
    src.stop()
    assert not src.alive
    n = len(sink.raws)
    assert sink.raws[:6] == [capture[i * FRAME : (i + 1) * FRAME] for i in (0, 1, 2, 3, 4, 0)]
    assert elapsed >= 11 / 25  # paced at 25 fps
    assert src.frames == n and src.rejected == 0
    time.sleep(0.1)
    assert len(sink.raws) == n  # nothing delivered after stop()


def test_replay_ends_without_frames(tmp_path: Path) -> None:
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\0" * 3 * FRAME)
    src = ReplaySource(junk, Sink())
    wait_for(lambda: not src.alive)
    assert src.frames == 0


def test_camera_source_streams_and_stops_001_AC1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture: bytes
) -> None:
    fake_v4l2(tmp_path, monkeypatch, f'exec tail -c +1 -f "{CAPTURE}"')  # stays open like a live camera
    sink = Sink()
    src = CameraSource("/dev/video9", sink)
    wait_for(lambda: len(sink.raws) == 5)
    assert src.alive and sink.raws[0] == capture[:FRAME]
    proc = src._proc
    src.stop()
    assert not src.alive and proc.returncode is not None
    assert src.error is None  # ended by us: no error


def test_camera_source_ends_when_v4l2_ctl_exits_001_AC8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_v4l2(tmp_path, monkeypatch, f'head -c {2 * FRAME + 100} "{CAPTURE}"; exit 1')  # unplug mid-frame
    sink = Sink()
    src: Source = CameraSource("/dev/video9", sink)
    wait_for(lambda: not src.alive)
    assert len(sink.raws) == 2
    src.stop()  # after the end: harmless


def test_watch_states_001_AC8() -> None:
    w = Watch(stall=3.0, poll=1.0)
    assert w.step(0.0, False, False) == Step.POLL  # first tick looks for the camera
    assert w.step(0.5, False, False) == Step.OK
    assert w.step(1.0, False, False) == Step.POLL  # not found: again 1 s later
    w.started(1.0)
    assert w.step(2.0, False, True) == Step.OK  # starting up, under the stall limit
    assert w.step(3.0, True, True) == Step.OK  # frames ok
    assert w.step(5.9, False, True) == Step.OK
    assert w.step(6.0, False, True) == Step.LOST  # stalled 3 s
    assert w.step(6.5, False, False) == Step.OK
    assert w.step(7.0, False, False) == Step.POLL  # gone: poll
    w.started(7.0)
    assert w.step(7.1, True, True) == Step.OK  # back
    assert w.step(7.2, True, False) == Step.OK  # last frame of a dying source still counts
    assert w.step(7.3, False, False) == Step.LOST  # source ended (v4l2-ctl exited)
    assert w.step(8.3, False, False) == Step.POLL


@pytest.mark.parametrize(("junk", "rejected"), [(1000, 0), (2 * FRAME, 2), (2 * FRAME + 1000, 2)])
def test_rejected_counts_skipped_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture: bytes, junk: int, rejected: int
) -> None:
    f = tmp_path / "gappy.bin"
    f.write_bytes(capture[: 2 * FRAME] + b"\xaa" * junk + capture[2 * FRAME :])
    fake_v4l2(tmp_path, monkeypatch, f'cat "{f}"; exec sleep 30')
    src = CameraSource("/dev/video9", Sink())
    wait_for(lambda: src.frames == 5)
    src.stop()
    assert src.rejected == rejected


def test_watch_defaults_001_AC8() -> None:
    w = Watch()
    assert (w.stall, w.poll) == (3.0, 1.0)


@pytest.mark.parametrize(
    ("stderr", "message"),
    [
        (b"", None),
        (b"<<<<<<<<<<", None),
        (b"<<<<\n<<<\n\n", None),
        (b"Cannot open device /dev/video9, exiting.\n", "Cannot open device /dev/video9, exiting."),
        (b"<<<<<<<<VIDIOC_DQBUF: failed: No such device\n<<\n", "VIDIOC_DQBUF: failed: No such device"),
        (b"first\nsecond\r\n  \n", "second"),
        (b"bad \xff byte\n<<<", "bad \ufffd byte"),
    ],
)
def test_last_message(stderr: bytes, message: str | None) -> None:
    assert last_message(stderr) == message


def test_camera_error_is_kept_001_AC8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = 'printf "<<<<\\n" >&2; echo "Cannot open device $2, exiting." >&2; exit 1'
    fake_v4l2(tmp_path, monkeypatch, script)
    src = CameraSource("/dev/video9", Sink())
    wait_for(lambda: not src.alive)
    assert src.error is None  # read once the source is stopped
    src.stop()
    assert src.error == "Cannot open device /dev/video9, exiting."
    src.stop()  # twice: harmless


@pytest.fixture
def cameras() -> Iterator[list[CameraSource]]:
    """CameraSources a test opens; their fake v4l2-ctl is SIGKILLed at teardown, whatever the test did."""
    opened: list[CameraSource] = []
    yield opened
    for src in opened:
        if src._proc.poll() is None:
            os.kill(src._proc.pid, signal.SIGKILL)  # not proc.kill(): a test may have stubbed it
            src._proc.wait(5)


def test_camera_stop_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cameras: list[CameraSource]
) -> None:
    """v4l2-ctl ignoring SIGTERM is killed; one that cannot be killed is abandoned, the UI goes on."""
    monkeypatch.setattr(source, "STOP_TIMEOUT", 0.5)
    pidfile = tmp_path / "pid"
    fake_v4l2(tmp_path, monkeypatch, f"trap '' TERM; echo $$ > {pidfile}; while :; do sleep 0.05; done")
    cameras.append(src := CameraSource("/dev/video9", Sink()))
    wait_for(pidfile.exists, timeout=30)
    t0 = time.monotonic()
    src.stop()
    assert time.monotonic() - t0 < 2 * 0.5 + 0.3
    assert src._proc.returncode == -9  # TERM ignored, then killed

    pidfile.unlink()  # before the second start, or its pid could be written first and then removed
    cameras.append(src := CameraSource("/dev/video9", Sink()))
    wait_for(pidfile.exists, timeout=30)
    proc = src._proc
    monkeypatch.setattr(proc, "kill", lambda: None)  # stands for a process stuck in the kernel
    t0 = time.monotonic()
    src.stop()
    assert time.monotonic() - t0 < 3 * 0.5 + 0.3
    assert proc.poll() is None  # abandoned, not waited for; the fixture kills it


@pytest.mark.parametrize(
    ("script", "error"),
    [
        ("exit 255", "v4l2-ctl exited with status 255 (camera busy or unreadable?)"),
        ('printf "<<<<" >&2; exit 3', "v4l2-ctl exited with status 3 (camera busy or unreadable?)"),
        ("exit 0", None),
        ("kill -SEGV $$", "v4l2-ctl killed by signal 11"),
        ('echo "real reason" >&2; exit 255', "real reason"),
    ],
)
def test_silent_exit_fallback_001_AC8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str, error: str | None
) -> None:
    fake_v4l2(tmp_path, monkeypatch, script)
    src = CameraSource("/dev/video9", Sink())
    wait_for(lambda: not src.alive)
    src.stop()
    assert src.error == error
