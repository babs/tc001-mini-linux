"""Photo (PNG with the Android tEXt JSON, raw .bin) and video (ffmpeg H.264 MP4) in the XDG folders."""

import fcntl
import os
import queue
import struct
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, tzinfo
from pathlib import Path

from tc001_linux import settings as cfg
from tc001_linux.picture import HEIGHT, WIDTH, format_celsius
from tc001_linux.process import Readouts
from tc001_linux.source import STDERR_TAIL, last_message

FOLDER = "TC001"
DEFAULT_DIRS = {"PICTURES": "Pictures", "VIDEOS": "Videos"}
XDG_TIMEOUT = 2.0
FPS = 25
QUEUE_FRAMES = 8  # ~0.3 s of pictures; past that the reader drops frames rather than wait on ffmpeg
FINALISE_TIMEOUT = 10.0  # after stop(): ffmpeg still busy then is killed and its file deleted


def capture_name(epoch_ms: int, tz: tzinfo | None = None) -> str:
    """`TC001_yyyyMMdd_HHmmss_SSS` in `tz` (default: local time), as the Android app's captureName."""
    t = _local(epoch_ms, tz)
    return f"TC001_{t:%Y%m%d_%H%M%S}_{epoch_ms % 1000:03d}"


def _local(epoch_ms: int, tz: tzinfo | None) -> datetime:
    # Whole seconds then the milliseconds: epoch_ms / 1000 as a float can round to the wrong millisecond.
    t = datetime.fromtimestamp(epoch_ms // 1000, UTC).astimezone(tz)
    return t.replace(microsecond=epoch_ms % 1000 * 1000)


def photo_json(r: Readouts, s: cfg.Settings, epoch_ms: int, tz: tzinfo | None = None) -> str:
    """The Android app's photoJson: same keys, order and formats, so photos from both read the same."""
    c = format_celsius
    t = _local(epoch_ms, tz).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    palette = s.palette.replace("-", "_")  # the Android enum names: white_hot, black_hot
    return (
        f'{{"min":{c(r.min)},"center":{c(r.centre)},"max":{c(r.max)},"lo":{c(r.lo)},"hi":{c(r.hi)},'
        f'"palette":"{palette}","rotation":{s.rotation},"mirror":{str(s.mirror).lower()},'
        f'"smoothing":{float(s.smoothing)!r},"locked":{str(r.locked).lower()},"time":"{t}"}}'
    )


def media_dir(kind: str, env: Mapping[str, str] = os.environ) -> Path:
    """TC001 folder in $XDG_<kind>_DIR, else `xdg-user-dir <kind>`, else ~/Pictures or ~/Videos."""
    home = Path(env.get("HOME") or Path.home())
    base = env.get(f"XDG_{kind}_DIR", "")
    if not os.path.isabs(base):
        try:
            run = subprocess.run(["xdg-user-dir", kind], capture_output=True, text=True, timeout=XDG_TIMEOUT)
            base = run.stdout.strip()
        except OSError, subprocess.SubprocessError:
            base = ""
        if base and Path(base) == home:  # xdg-user-dir answers $HOME for a folder it has no entry for
            base = ""
    return (Path(base) if os.path.isabs(base) else home / DEFAULT_DIRS[kind]) / FOLDER


def _write(path: Path, data: bytes) -> None:
    with path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())  # on disk before the rename, or a power cut can leave an empty file


def save_photo(folder: Path, name: str, png: bytes, raw: bytes) -> Path:
    """Writes `name`.png and `name`.bin in `folder`, both or neither; returns the PNG path."""
    folder.mkdir(parents=True, exist_ok=True)
    final = [folder / f"{name}.png", folder / f"{name}.bin"]
    parts = [folder / f".{p.name}.part" for p in final]
    try:
        for part, data in zip(parts, (png, raw), strict=True):
            _write(part, data)
        for part, path in zip(parts, final, strict=True):
            os.replace(part, path)
    except BaseException:
        for p in (*parts, *final):
            p.unlink(missing_ok=True)
        raise
    return final[0]


def record_command(out: Path) -> list[str]:
    # Wall-clock input timestamps resampled to a constant 25 fps: dropped or late frames keep the real
    # duration (the last picture is repeated), and players get a fixed frame rate.
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-use_wallclock_as_timestamps",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        # RGB to BT.709 limited range, and the tags that say so: untagged, players guess BT.601 at this size.
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv,setparams=color_primaries=bt709:color_trc=bt709",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-fps_mode",
        "cfr",
        "-r",
        str(FPS),
        "-f",
        "mp4",
        "-y",
        str(out),
    ]


def has_moov(path: Path) -> bool:
    """The MP4 holds its whole index (the moov box ffmpeg writes when it finalises): else nothing plays."""
    try:
        with path.open("rb") as f:
            end, pos = os.fstat(f.fileno()).st_size, 0
            while pos + 8 <= end:
                f.seek(pos)
                size, kind = struct.unpack(">I4s", f.read(8))
                if size == 1:  # 64-bit size follows the type
                    if pos + 16 > end:
                        return False
                    (size,) = struct.unpack(">Q", f.read(8))
                # 0 = box runs to the end of the file (never a finished index); < 8 = corrupt.
                if size < 8 or pos + size > end:
                    return False
                if kind == b"moov":
                    return True
                pos += size
    except OSError:
        pass
    return False


def sweep(folder: Path) -> tuple[list[str], list[str]]:
    """Leftovers of a crash in `folder`: a video part holding its index becomes the video, other parts go.

    Returns (recovered video names, discarded part names). A part locked by a live Recorder is left alone.
    """
    recovered, discarded = [], []
    for part in folder.glob(".TC001_*.part"):
        final = folder / part.name[1:].removesuffix(".part")
        try:
            with part.open("rb") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue  # being recorded, possibly by another viewer
                if final.suffix == ".mp4" and not final.exists() and has_moov(part):
                    os.replace(part, final)
                    recovered.append(final.name)
                else:  # a photo part: its pair may be missing, so both-or-neither means none
                    part.unlink()
                    discarded.append(part.name)
        except OSError:
            continue
    return recovered, discarded


class Recorder:
    """H.264 MP4 of the pictures given to feed(), through ffmpeg; a writer thread feeds ffmpeg's stdin.

    Finalised on stop() or when ffmpeg ends; kept as `path` only if a frame went in and the file has its
    index, else deleted. `done` gets the outcome message, on the writer thread.
    """

    def __init__(self, folder: Path, name: str, done: Callable[[str], None]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{name}.mp4"
        self._part = folder / f".{name}.mp4.part"
        self._done = done
        self.frames = self.dropped = 0
        self._queue: queue.Queue[bytes] = queue.Queue(QUEUE_FRAMES)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stopping = False
        self._killed = False
        self._timer = threading.Timer(FINALISE_TIMEOUT, self._kill)
        self._timer.daemon = True
        # Held for the recorder's life (ffmpeg's -y truncates, the inode and its lock stay): sweep() skips it.
        self._part_lock = self._part.open("wb")
        self._stderr = tempfile.TemporaryFile()
        try:
            fcntl.flock(self._part_lock, fcntl.LOCK_EX)
            # Own process group: a Ctrl-C in the terminal reaches only us, and we finalise the file.
            # ffmpeg inherits the locked fd: if we die, the lock lasts until it has finished the file.
            self._proc = subprocess.Popen(
                record_command(self._part),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
                process_group=0,
                pass_fds=(self._part_lock.fileno(),),
            )
        except BaseException:
            self._stderr.close()
            self._part.unlink(missing_ok=True)
            self._part_lock.close()
            raise
        # Not a daemon: the interpreter waits for the file to be finalised before it exits.
        self._thread = threading.Thread(target=self._run, name="recorder")
        self._thread.start()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def feed(self, rgb: bytes) -> None:
        """Reader thread: queues one WIDTH x HEIGHT rgb24 picture; dropped if ffmpeg lags behind."""
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(rgb)
        except queue.Full:
            self.dropped += 1

    def stop(self) -> None:
        """Any thread, never blocks: ends the input; ffmpeg gets FINALISE_TIMEOUT s before it is killed."""
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
        self._stop.set()
        self._timer.start()  # a cancel() that came first is kept: the timer then never fires

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _kill(self) -> None:
        # Also frees a writer blocked in write() on an ffmpeg that stopped reading.
        if self._thread.is_alive():
            self._killed = True
            self._proc.kill()

    def _run(self) -> None:
        stdin = self._proc.stdin
        assert stdin is not None
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                rgb = self._queue.get(timeout=0.05)
            except queue.Empty:
                if self._proc.poll() is not None:  # ffmpeg ended while no frame came
                    break
                continue
            try:
                stdin.write(rgb)
            except OSError:  # ffmpeg ended
                break
            self.frames += 1
        self.stop()
        try:
            stdin.close()  # ffmpeg sees the end of input and writes the index
        except OSError:
            pass
        rc = self._proc.wait()  # bounded by the stop() timer
        self._timer.cancel()  # else it holds this recorder and the window behind `done` for 10 s
        self._stderr.seek(max(0, self._stderr.seek(0, os.SEEK_END) - STDERR_TAIL))
        error = last_message(self._stderr.read())
        self._stderr.close()
        message = None
        # A non-zero exit can still be a finalised file: ffmpeg writes the index on SIGTERM / SIGINT.
        if self.frames and has_moov(self._part):
            try:
                os.replace(self._part, self.path)
                message = f"Video saved: {self.path.name}"
            except OSError as e:
                error = str(e)
        if message is None:
            self._part.unlink(missing_ok=True)
            if self._killed:
                error = f"ffmpeg did not finish within {FINALISE_TIMEOUT:g} s"
            elif not self.frames and rc == 0:
                error = "no frame recorded"
            message = f"Video failed: {error or f'ffmpeg exited with status {rc}'}"
        self._part_lock.close()  # after the rename or the unlink: a sweep never sees the part unlocked
        self._done(message)  # outside the OSError handling: a failing `done` cannot turn saved into failed
