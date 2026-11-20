"""Frame sources (camera through v4l2-ctl, looped capture), latest-frame mailbox, link watchdog."""

import enum
import math
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from tc001_linux import Plane
from tc001_linux.frame import FRAME, Readable, frames, parse

VID_PID = ("2bdf", "0102")
SYSFS = Path("/sys/class/video4linux")
REPLAY_FPS = 25.0
STOP_TIMEOUT = 2.0
STDERR_TAIL = 4096

type OnFrame = Callable[[bytes, Plane], None]


def find_device(sysfs: Path = SYSFS) -> str | None:
    """/dev node of the TC001 Mini's streaming interface (V4L2 index 0 of USB 2bdf:0102), or None."""
    for node in sorted(sysfs.glob("video*")):
        usb = node / "device" / ".."  # device -> the USB interface; its parent holds the USB ids
        try:
            ids = ((usb / "idVendor").read_text().strip(), (usb / "idProduct").read_text().strip())
            if ids == VID_PID and (node / "index").read_text().strip() == "0":
                return f"/dev/{node.name}"
        except OSError:
            continue
    return None


def camera_command(device: str) -> list[str]:
    return [
        "v4l2-ctl",
        "-d",
        device,
        "--set-fmt-video=width=4,height=2321,pixelformat=YUYV",
        "--stream-mmap",
        "--stream-to=-",
    ]


def last_message(stderr: bytes) -> str | None:
    """Last line of v4l2-ctl's stderr once its '<' frame ticks are dropped, or None."""
    for line in reversed(stderr.decode(errors="replace").splitlines()):
        if text := line.lstrip("<").strip():
            return text
    return None


class Mailbox[T]:
    """One-slot hand-over between threads: put() replaces an untaken item, so the reader never waits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None
        self.dropped = 0

    def put(self, item: T) -> None:
        with self._lock:
            if self._item is not None:
                self.dropped += 1
            self._item = item

    def take(self) -> T | None:
        with self._lock:
            item, self._item = self._item, None
            return item


class _Counted:
    def __init__(self, raw: Readable) -> None:
        self.raw, self.n = raw, 0

    def read(self, size: int, /) -> bytes:
        b = self.raw.read(size)
        self.n += len(b)
        return b


class Source:
    """Reader thread handing every parsed frame to `on_frame`; alive until its stream ends or stop()."""

    period = 0.0  # seconds between frames; 0 = as fast as the stream delivers

    def __init__(self, on_frame: OnFrame) -> None:
        self.on_frame = on_frame
        self.frames = 0
        self.error: str | None = None  # why the stream ended, once stopped
        self._bad = self._read = self._chunks = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def rejected(self) -> int:
        # ponytail: bytes skipped by the resync, in frames; off by one while a partial frame is buffered.
        return self._bad + max(0, self._read - FRAME * self._chunks) // FRAME

    def _run(self) -> None:
        raise NotImplementedError

    def _feed(self, stream: Readable) -> bool:
        """Deliver the frames of `stream` until it ends or stop(); True if it held at least one."""
        counted, start, got = _Counted(stream), self._read, False
        next_at = time.monotonic()
        for raw in frames(counted):
            self._read, self._chunks = start + counted.n, self._chunks + 1
            if self.period:
                now = time.monotonic()
                next_at = max(next_at + self.period, now - 1.0)  # no burst to catch up after a stall
                if self._stop.wait(max(0.0, next_at - now)):
                    return got
            if self._stop.is_set():
                return got
            t = parse(raw)
            if t is None:
                self._bad += 1
                continue
            self.on_frame(raw, t)
            self.frames += 1
            got = True
        self._read = start + counted.n
        return got

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(STOP_TIMEOUT)


class ReplaySource(Source):
    """A raw mode-2 capture, looped at 25 fps; ends if a whole pass holds no frame."""

    period = 1.0 / REPLAY_FPS

    def __init__(self, path: Path, on_frame: OnFrame) -> None:
        super().__init__(on_frame)
        self.path = path
        self._thread.start()

    def _run(self) -> None:
        with self.path.open("rb") as f:
            while self._feed(f) and not self._stop.is_set():
                f.seek(0)


class CameraSource(Source):
    """The camera through `v4l2-ctl` in mode 2; ends when v4l2-ctl exits (unplug, busy device)."""

    def __init__(self, device: str, on_frame: OnFrame) -> None:
        super().__init__(on_frame)
        self.device = device
        # A file, not a pipe: v4l2-ctl writes a '<' per frame and would block once a pipe filled up.
        self._stderr = tempfile.TemporaryFile()
        null, pipe = subprocess.DEVNULL, subprocess.PIPE
        self._proc = subprocess.Popen(camera_command(device), stdin=null, stdout=pipe, stderr=self._stderr)
        self._thread.start()

    def _run(self) -> None:
        assert self._proc.stdout is not None
        self._feed(self._proc.stdout)

    def stop(self) -> None:
        """Ends v4l2-ctl and the reader within ~3 x STOP_TIMEOUT, so the window never freezes."""
        if self._stop.is_set():
            return
        self._stop.set()
        exited = None  # v4l2-ctl's own exit status, before we signal it
        if not self._thread.is_alive():  # stream ended: v4l2-ctl is exiting by itself
            try:
                exited = self._proc.wait(STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
        for end in (self._proc.terminate, self._proc.kill):
            end()
            try:
                self._proc.wait(STOP_TIMEOUT)
                break
            except subprocess.TimeoutExpired:
                pass  # ponytail: a process stuck in the kernel is abandoned; it ends when the device lets go
        self._thread.join(STOP_TIMEOUT)
        assert self._proc.stdout is not None
        if not self._thread.is_alive():  # a reader blocked in read() holds the buffer lock: close() hangs
            self._proc.stdout.close()
        size = self._stderr.seek(0, os.SEEK_END)
        self._stderr.seek(max(0, size - STDERR_TAIL))
        self.error = last_message(self._stderr.read())
        if self.error is None and exited is not None and exited < 0:
            self.error = f"v4l2-ctl killed by signal {-exited}"
        elif self.error is None and exited:  # a busy device: v4l2-ctl exits 255 and prints nothing
            self.error = f"v4l2-ctl exited with status {exited} (camera busy or unreadable?)"
        self._stderr.close()


class Step(enum.Enum):
    OK = enum.auto()
    LOST = enum.auto()  # stop the source, show "Camera disconnected"
    POLL = enum.auto()  # look for the device; call started() if a source was opened


class Watch:
    """Link state: lost after `stall` s with no frame or once the source ends; then a poll every `poll` s."""

    def __init__(self, stall: float = 3.0, poll: float = 1.0) -> None:
        self.stall, self.poll = stall, poll
        self.running = False
        self._last = 0.0
        self._next_poll = -math.inf

    def started(self, now: float) -> None:
        self.running, self._last = True, now

    def step(self, now: float, got_frame: bool, alive: bool) -> Step:
        if self.running:
            if got_frame:
                self._last = now
                return Step.OK
            if alive and now - self._last < self.stall:
                return Step.OK
            self.running, self._next_poll = False, now + self.poll
            return Step.LOST
        if now >= self._next_poll:
            self._next_poll = now + self.poll
            return Step.POLL
        return Step.OK
