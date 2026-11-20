"""Window smoke test on a private Xvfb; skipped without Xvfb locally, failed without it in CI (001-AC9)."""

import gc
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import pytest

from tc001_linux import settings as cfg
from tc001_linux.capture import capture_name
from tc001_linux.frame import FRAME, parse
from tc001_linux.picture import (
    BAR_MARGIN,
    BAR_TOP,
    BAR_WIDTH,
    HEIGHT,
    WIDTH,
    bar_row,
    format_celsius,
    readout_texts,
)
from tc001_linux.process import Readouts
from tc001_linux.settings import Settings
from tc001_linux.source import CameraSource, OnFrame, ReplaySource, Source

from .conftest import CAPTURE, decode_png, fake_exe, fake_v4l2, ffprobe, png_chunks

try:
    import tkinter
    import tkinter.ttk
except ImportError:
    if os.environ.get("CI"):
        raise
    pytest.skip("no tkinter: install python3-tk, the system Python's Tk", allow_module_level=True)


@pytest.fixture(scope="session")
def x_server() -> Iterator[str]:
    """One Xvfb for the session: Tk keeps its X connection after destroy(), and dies if that server goes."""
    if shutil.which("Xvfb") is None:
        msg = "Xvfb not installed: the window test needs an X server (apt install xvfb)"
        if os.environ.get("CI"):
            pytest.fail(msg)
        pytest.skip(msg)
    r, w = os.pipe()
    cmd = ["Xvfb", "-displayfd", str(w), "-screen", "0", "1024x1280x24", "-nolisten", "tcp"]
    proc = subprocess.Popen(cmd, pass_fds=(w,), stderr=subprocess.DEVNULL)
    os.close(w)
    with os.fdopen(r) as f:
        display = f.readline().strip()
    try:
        assert display, "Xvfb did not start"
        yield f":{display}"
    finally:
        proc.terminate()
        proc.wait(5)


@pytest.fixture
def xvfb(x_server: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DISPLAY", x_server)
    yield
    # A test's Tk images die here, on the main thread: collected later on a reader thread, Tk refuses.
    gc.collect()


def pump(root: tkinter.Tk, cond: Callable[[], bool], timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        root.update()
        time.sleep(0.005)


def pixels(ppm: bytes) -> bytes:
    """RGB payload of a P6 image, whatever whitespace its header uses."""
    return ppm[len(b" ".join(ppm.split(maxsplit=4)[:4])) + 1 :]


def test_window_on_capture_001_AC4_AC5_AC9(xvfb: None, tmp_path: Path) -> None:
    import tkinter as tk

    from tc001_linux.window import App

    root = tk.Tk()
    path = tmp_path / "settings.json"
    sources: list[ReplaySource] = []

    def connect(on_frame: OnFrame) -> ReplaySource:
        sources.append(ReplaySource(CAPTURE, on_frame))
        return sources[-1]

    app = App(root, connect, Settings(), path)

    try:
        pump(root, lambda: app.readout_shot is not None)
        shot = app.readout_shot
        assert shot is not None
        # Readouts: the cold / hot marker and centre badge values of the frame on screen at that moment.
        r = shot.readouts
        badges = (r.cold.celsius, r.hot.celsius, r.centre)
        assert app.readout_texts == tuple(format_celsius(v) for v in badges)
        values = (app.min_label, app.max_label, app.centre_label)
        assert tuple(str(v.cget("text")) for v in values) == app.readout_texts
        assert [str(v.cget("foreground")) for v in values] == ["#00c8ff", "#ff4d6d", "#ffffff"]
        assert app.readout_fonts[2].cget("size") < app.readout_fonts[1].cget("size") < 0  # px: Center larger
        assert pixels(app.photo.data(format="ppm")) == shot.picture.tobytes()

        first, shown = app.photo.data(format="ppm"), app.shown
        # The picture updates.
        pump(root, lambda: app.shown >= shown + 3 and app.photo.data(format="ppm") != first)

        root.focus_force()
        root.update()
        root.event_generate("<KeyPress>", keysym="c")
        assert app.settings.palette == "white-hot"
        assert json.loads(path.read_text())["palette"] == "white-hot"
        assert app.palette_button.cget("text") == "White hot"
        mid = BAR_MARGIN + BAR_WIDTH // 2
        want = tuple(int(c) for c in bar_row("white-hot")[mid - BAR_MARGIN])
        assert want != tuple(int(c) for c in bar_row("ironbow")[mid - BAR_MARGIN])
        # Waits on the pixel, not a frame count: a frame composed before the key cannot pass.
        pump(root, lambda: tuple(app.photo.get(mid, BAR_TOP + 1)) == want)

        root.event_generate("<KeyPress>", keysym="l")
        assert app.proc.locked and app.lock_button.cget("text") == "Range locked"
        root.event_generate("<KeyPress>", keysym="plus")
        assert app.smoothing.get() == pytest.approx(0.6)
        assert json.loads(path.read_text())["smoothing"] == pytest.approx(0.6)
        assert int(root.winfo_width()) == WIDTH
    finally:
        app.quit()
    assert sources and not any(s.alive for s in sources)


def test_window_loses_and_finds_the_source_001_AC8(xvfb: None, tmp_path: Path) -> None:
    import tkinter as tk

    from tc001_linux.window import DISCONNECTED, App

    root = tk.Tk()
    capture = tmp_path / "c.bin"
    capture.write_bytes(CAPTURE.read_bytes())
    sources: list[ReplaySource] = []

    def connect(on_frame: OnFrame) -> ReplaySource | None:
        if not capture.exists():
            return None
        sources.append(ReplaySource(capture, on_frame))
        return sources[-1]

    app = App(root, connect, Settings(), tmp_path / "settings.json", debug=True)
    try:
        pump(root, lambda: app.readout_texts is not None)
        assert not app.message.winfo_ismapped()
        capture.unlink()
        sources[-1].stop()  # the reader ends, as when v4l2-ctl exits on unplug
        pump(root, lambda: app.source is None)
        root.update()
        assert app.message.winfo_ismapped() and app.message.cget("text") == DISCONNECTED
        assert app.readout_texts is None and not any(pixels(app.photo.data(format="ppm")))
        assert app.centre_label.cget("text") == ""
        shown = app.shown
        for _ in range(150):  # 1.5 s: polled, nothing there yet
            root.update()
            time.sleep(0.01)
        assert app.source is None and app.shown == shown and len(sources) == 1
        capture.write_bytes(CAPTURE.read_bytes())
        t0 = time.monotonic()
        pump(root, lambda: app.shown > shown and app.readout_texts is not None)
        assert time.monotonic() - t0 < 5 and not app.message.winfo_ismapped() and len(sources) == 2
        debug = app.debug
        assert debug is not None
        pump(root, lambda: "fps" in str(debug.cget("text")))
    finally:
        app.quit()
    assert not any(s.alive for s in sources)


def test_frame_landing_during_stop_is_dropped_001_AC8(xvfb: None, tmp_path: Path) -> None:
    import tkinter as tk

    from tc001_linux.window import DISCONNECTED, App

    raw = CAPTURE.read_bytes()[:FRAME]

    class Dying(Source):
        """Ended at once; a last frame reaches the mailbox while stop() joins the reader."""

        def stop(self) -> None:
            t = parse(raw)
            assert t is not None
            self.on_frame(raw, t)

    root = tk.Tk()
    opened: list[Source] = []

    def connect(on_frame: OnFrame) -> Source | None:
        if opened:
            return None
        opened.append(Dying(on_frame))
        return opened[-1]

    app = App(root, connect, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: app.source is None and bool(opened))
        for _ in range(20):
            root.update()
            time.sleep(0.01)
        assert app.shown == 0 and app.message.cget("text") == DISCONNECTED and app.message.winfo_ismapped()
    finally:
        app.quit()


def test_source_open_error_keeps_polling(
    xvfb: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import tkinter as tk

    from tc001_linux.window import App

    calls: list[float] = []

    def connect(on_frame: OnFrame) -> Source | None:
        calls.append(time.monotonic())
        raise PermissionError("v4l2-ctl: permission denied")

    root = tk.Tk()
    app = App(root, connect, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: len(calls) >= 2)  # the tick loop survived the first failure
    finally:
        app.quit()
    assert "cannot open the source: v4l2-ctl: permission denied" in capsys.readouterr().err


def replay(on_frame: OnFrame) -> Source:
    return ReplaySource(CAPTURE, on_frame)


def test_slider_drag_saves_once_001_AC5(xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tc001_linux.window import App

    saves: list[Settings] = []
    monkeypatch.setattr(cfg, "save", lambda p, s: saves.append(s))
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: app.scale.winfo_width() > 100)
        sc = app.scale
        x, y = sc.coords(0.5)
        sc.event_generate("<ButtonPress-1>", x=x, y=y)
        seen = set()
        for dx in range(4, 120, 4):
            sc.event_generate("<B1-Motion>", x=x + dx, y=y)
            root.update()
            seen.add(app.settings.smoothing)
        assert len(seen) > 5 and not saves  # applied live, not written yet
        sc.event_generate("<ButtonRelease-1>", x=x + 120, y=y)
        root.update()
        assert saves == [app.settings] and app.settings.smoothing > 0.6
        root.event_generate("<KeyPress>", keysym="minus")  # a key still saves at once
        assert len(saves) == 2 and saves[-1] == app.settings
    finally:
        app.quit()
    assert len(saves) == 2  # nothing left to flush on quit


def test_fast_drag_saves_the_last_value_001_AC5(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Motions and release queued together: the release saves the last value, once."""
    from tc001_linux.window import App

    saves: list[Settings] = []
    monkeypatch.setattr(cfg, "save", lambda p, s: saves.append(s))
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: app.scale.winfo_width() > 100)
        sc = app.scale
        x, y = sc.coords(0.5)
        sc.event_generate("<ButtonPress-1>", x=x, y=y, when="tail")
        for dx in range(10, 160, 10):
            sc.event_generate("<B1-Motion>", x=x + dx, y=y, when="tail")
        sc.event_generate("<ButtonRelease-1>", x=x + 150, y=y, when="tail")
        root.update()
        assert sc.get() > 0.7 and app.settings.smoothing == sc.get()
        assert saves == [app.settings]
    finally:
        app.quit()


def test_quit_saves_an_unreleased_slider_001_AC5(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tc001_linux.window import App

    saves: list[Settings] = []
    monkeypatch.setattr(cfg, "save", lambda p, s: saves.append(s))
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        app.scale.set(0.8)  # moved, never released
        root.update()
        assert app.settings.smoothing == 0.8 and not saves
    finally:
        app.quit()
    assert saves == [Settings(smoothing=0.8)]


@pytest.mark.parametrize(
    ("script", "error"),
    [
        (
            'printf "<<<" >&2; echo "Cannot open device $2, exiting." >&2; exit 1',
            "Cannot open device /dev/video9, exiting.",
        ),
        ("exit 255", "v4l2-ctl exited with status 255 (camera busy or unreadable?)"),
    ],
)
def test_source_error_shown_and_printed_once_001_AC8(
    xvfb: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: str,
    error: str,
) -> None:
    from tc001_linux.window import DISCONNECTED, App

    fake_v4l2(tmp_path, monkeypatch, script)
    opened: list[Source] = []

    def connect(on_frame: OnFrame) -> Source:
        opened.append(CameraSource("/dev/video9", on_frame))
        return opened[-1]

    root = tkinter.Tk()
    app = App(root, connect, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: len(opened) >= 3 and app.source is None)
        assert app.message.cget("text") == f"{DISCONNECTED}\n{error}"
    finally:
        app.quit()
    assert capsys.readouterr().err.count(error) == 1  # once per distinct message


def test_fixed_range_lock_001_AC5(xvfb: None, tmp_path: Path) -> None:
    from tc001_linux.window import App

    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json", fixed=(0.0, 50.0))
    try:
        assert app.lock_button.cget("text") == "Fixed range" and app.lock_button.instate(["disabled"])
        app.act("lock")
        assert not app.proc.locked and app.lock_button.cget("text") == "Fixed range"
    finally:
        app.quit()


def test_unwritable_settings_keep_the_view(
    xvfb: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tc001_linux.window import App

    (tmp_path / "file").touch()
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "file" / "settings.json")
    try:
        pump(root, lambda: app.shown > 0)
        root.focus_force()
        root.update()
        root.event_generate("<KeyPress>", keysym="c")
        assert app.settings.palette == "white-hot"
        assert "tc001-mini-linux: settings not saved:" in capsys.readouterr().err
        shown = app.shown
        pump(root, lambda: app.shown > shown + 2)  # the loop goes on
    finally:
        app.quit()


def test_readouts_refresh_about_5_hz_001_AC4(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tc001_linux import window
    from tc001_linux.window import App

    refreshed: list[float] = []

    def counting(r: Readouts) -> tuple[str, str, str]:
        refreshed.append(time.monotonic())
        return readout_texts(r)

    monkeypatch.setattr(window, "readout_texts", counting)
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: bool(refreshed) and time.monotonic() - refreshed[0] >= 2.0)
    finally:
        app.quit()
    gaps = sorted(b - a for a, b in itertools.pairwise(refreshed))
    median = gaps[len(gaps) // 2]
    assert 0.15 <= median <= 0.25, gaps  # READOUT_PERIOD 0.2 s, waiting for the next 40 ms frame


def test_run_stops_the_source_when_the_loop_ends(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tc001_linux.window import run

    calls: list[tuple[int, object]] = []

    def fake_signal(sig: int, handler: object) -> object:
        calls.append((sig, handler))
        return f"previous {sig}"

    monkeypatch.setattr(signal, "signal", fake_signal)
    opened: list[Source] = []

    def connect(on_frame: OnFrame) -> Source:
        root = tkinter._default_root  # type: ignore[attr-defined]
        root.after(300, root.quit)  # the Tk loop ends without App.quit()
        opened.append(ReplaySource(CAPTURE, on_frame))
        return opened[-1]

    try:
        run(connect, Settings(), tmp_path / "settings.json")
        sigs = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
        assert [sig for sig, _ in calls[:3]] == sigs
        assert calls[3:] == [(sig, f"previous {sig}") for sig in sigs]  # restored
        assert len(opened) == 1 and not opened[0].alive
    finally:
        for src in opened:
            src.stop()
        root = tkinter._default_root  # type: ignore[attr-defined]
        if root is not None:
            for pending in root.tk.splitlist(root.tk.call("after", "info")):
                root.after_cancel(pending)  # the tick App.quit() would have cancelled
            root.destroy()


def test_sigterm_quits_cleanly_001_AC8(
    x_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real program on the camera path: SIGTERM ends it with rc 0 and v4l2-ctl gone."""
    pidfile, device = tmp_path / "v4l2.pid", tmp_path / "video9"
    device.touch()
    fake_v4l2(tmp_path, monkeypatch, f'echo $$ > {pidfile}; exec tail -c +1 -f "{CAPTURE}"')
    env = dict(os.environ, DISPLAY=x_server, XDG_CONFIG_HOME=str(tmp_path / "cfg"))
    code = "from tc001_linux.cli import main; main()"
    app = subprocess.Popen([sys.executable, "-c", code, "--device", str(device)], env=env)
    child = None
    try:
        end = time.monotonic() + 15
        while not (pidfile.exists() and pidfile.read_text().strip()):
            assert app.poll() is None and time.monotonic() < end, "v4l2-ctl never started"
            time.sleep(0.05)
        child = int(pidfile.read_text())
        time.sleep(0.3)
        app.send_signal(signal.SIGTERM)
        assert app.wait(5) == 0
        end = time.monotonic() + 2
        while True:
            try:
                if open(f"/proc/{child}/stat").read().split()[2] == "Z":
                    break  # dead, not yet reaped
            except FileNotFoundError:
                break  # gone
            assert time.monotonic() < end, "v4l2-ctl left running"
            time.sleep(0.05)
    finally:
        if app.poll() is None:
            app.kill()
            app.wait(5)
        if child is not None and os.path.exists(f"/proc/{child}"):
            os.kill(child, signal.SIGKILL)


def video_check(path: Path, wall: float) -> None:
    """Playable H.264 768x832 at 25 fps, no audio, as long as the recording within 1 s."""
    info = ffprobe(path)
    streams = info["streams"]
    assert isinstance(streams, list) and [s["codec_type"] for s in streams] == ["video"]
    v = streams[0]
    assert (v["codec_name"], v["width"], v["height"], v["avg_frame_rate"]) == ("h264", 768, 832, "25/1")
    duration = float(info["format"]["duration"])  # type: ignore[index]
    assert abs(duration - wall) < 1.0, (duration, wall)


def saved(app: object, root: tkinter.Tk, prefix: str, timeout: float = 15) -> str:
    """Waits for a status line starting with `prefix`; returns the rest."""
    status = app.status  # type: ignore[attr-defined]
    pump(root, lambda: str(status.cget("text")).startswith(prefix), timeout)
    return str(status.cget("text")).removeprefix(prefix)


def test_photo_001_AC6(xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tc001_linux import window
    from tc001_linux.window import App, Shot

    monkeypatch.setenv("XDG_PICTURES_DIR", str(tmp_path / "pics"))
    monkeypatch.setattr(window, "STATUS_SECONDS", 0.5)
    root = tkinter.Tk()
    s = Settings("black-hot", 90, True, 0.3)
    app = App(root, replay, s, tmp_path / "settings.json")
    shots: list[Shot] = []
    put = app.mailbox.put

    def spy(shot: Shot) -> None:
        shots.append(shot)
        put(shot)

    monkeypatch.setattr(app.mailbox, "put", spy)
    folder = tmp_path / "pics" / "TC001"
    try:
        pump(root, lambda: app.shown > 0)
        root.focus_force()
        root.update()
        root.event_generate("<KeyPress>", keysym="p")
        name = saved(app, root, "Photo saved: ")
        assert sorted(p.name for p in folder.iterdir()) == [f"{name}.bin", f"{name}.png"]
        png, raw = (folder / f"{name}.png").read_bytes(), (folder / f"{name}.bin").read_bytes()
        capture = CAPTURE.read_bytes()
        assert len(raw) == 18_568 and raw in {capture[i : i + FRAME] for i in range(0, len(capture), FRAME)}
        img = decode_png(png)
        assert img.shape == (832, 768, 3)
        kind, text = png_chunks(png)[1]
        keyword, _, body = text.partition(b"\0")
        assert (kind, keyword) == (b"tEXt", b"tc001")
        meta = json.loads(body.decode("latin-1"))
        # The frame of the .bin: same picture, and its readouts are the JSON's.
        frame = [sh for sh in shots if sh.raw == raw and sh.picture.tobytes() == img.tobytes()]
        assert frame, "no frame with this raw and this picture"
        r = frame[0].readouts
        assert meta == {
            "min": float(format_celsius(r.min)),
            "center": float(format_celsius(r.centre)),
            "max": float(format_celsius(r.max)),
            "lo": float(format_celsius(r.lo)),
            "hi": float(format_celsius(r.hi)),
            "palette": "black_hot",
            "rotation": 90,
            "mirror": True,
            "smoothing": 0.3,
            "locked": False,
            "time": meta["time"],
        }
        ms = int(datetime.fromisoformat(meta["time"]).timestamp() * 1000 + 0.5)
        assert capture_name(ms) == name
        pump(root, lambda: app.status.cget("text") == "")  # the status line clears
        app.photo_button.invoke()  # the button takes one too
        second = saved(app, root, "Photo saved: ")
        assert second != name and len(list(folder.iterdir())) == 4
    finally:
        app.quit()


def test_photo_failure_leaves_nothing_001_AC6(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tc001_linux.window import App

    (tmp_path / "file").touch()
    monkeypatch.setenv("XDG_PICTURES_DIR", str(tmp_path / "file"))
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: app.shown > 0)
        app.act("photo")
        error = saved(app, root, "Photo failed: ")
        assert "Not a directory" in error
        assert sorted(p.name for p in tmp_path.iterdir()) == ["file"]
        assert f"Photo failed: {error}" in capsys.readouterr().err
    finally:
        app.quit()


def start_recording(app: object, root: tkinter.Tk) -> float:
    pump(root, lambda: app.shown > 0)  # type: ignore[attr-defined]
    root.focus_force()
    root.update()
    root.event_generate("<KeyPress>", keysym="r")
    from tc001_linux.window import RED

    button = app.record_button  # type: ignore[attr-defined]
    style = str(button.cget("style"))
    assert button.cget("text") == "Stop" and tkinter.ttk.Style(root).lookup(style, "foreground") == RED
    return time.monotonic()


def test_record_stop_001_AC7(xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tc001_linux.capture import media_dir
    from tc001_linux.window import App

    monkeypatch.setenv("XDG_VIDEOS_DIR", str(tmp_path / "vids"))
    resolved: list[str] = []

    def counted(kind: str) -> Path:
        resolved.append(kind)
        return media_dir(kind)

    monkeypatch.setattr("tc001_linux.window.media_dir", counted)
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        t0 = start_recording(app, root)
        rec = app.recorder
        assert rec is not None
        pump(root, lambda: time.monotonic() - t0 >= 4.0)
        app.record_button.invoke()  # Stop
        wall = time.monotonic() - t0
        assert app.recorder is None and app.record_button.cget("text") == "Record"
        assert app.record_button.cget("style") == "TButton"
        name = saved(app, root, "Video saved: ")
        assert rec.path.name == name and list((tmp_path / "vids" / "TC001").iterdir()) == [rec.path]
        video_check(rec.path, wall)
        app.act("photo")
        saved(app, root, "Photo saved: ")
        assert resolved == ["PICTURES", "VIDEOS"]  # once, at start: xdg-user-dir never runs on a press
    finally:
        app.quit()


def test_quit_finalises_the_recording_001_AC7(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tc001_linux.window import App

    monkeypatch.setenv("XDG_VIDEOS_DIR", str(tmp_path))
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    t0 = start_recording(app, root)
    rec = app.recorder
    assert rec is not None
    pump(root, lambda: time.monotonic() - t0 >= 2.0)
    root.event_generate("<KeyPress>", keysym="q")
    wall = time.monotonic() - t0
    rec.join(15)
    assert list((tmp_path / "TC001").iterdir()) == [rec.path]
    video_check(rec.path, wall)


def test_source_loss_finalises_the_recording_001_AC7(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tc001_linux.window import App

    monkeypatch.setenv("XDG_VIDEOS_DIR", str(tmp_path / "vids"))
    capture = tmp_path / "c.bin"
    capture.write_bytes(CAPTURE.read_bytes())
    sources: list[ReplaySource] = []

    def connect(on_frame: OnFrame) -> ReplaySource | None:
        if not capture.exists():
            return None
        sources.append(ReplaySource(capture, on_frame))
        return sources[-1]

    root = tkinter.Tk()
    app = App(root, connect, Settings(), tmp_path / "settings.json")
    try:
        t0 = start_recording(app, root)
        rec = app.recorder
        assert rec is not None
        pump(root, lambda: time.monotonic() - t0 >= 2.0)
        capture.unlink()
        sources[-1].stop()  # the camera goes
        wall = time.monotonic() - t0
        pump(root, lambda: app.source is None)
        assert app.recorder is None and app.record_button.cget("text") == "Record"
        app.act("record")  # nothing to record without a source
        app.act("photo")
        assert app.recorder is None and not app._photo.is_set()
        saved(app, root, "Video saved: ")
        video_check(rec.path, wall)  # stops at the last frame; the stall detection adds no time
    finally:
        app.quit()


def test_ffmpeg_failure_resets_the_button_001_AC7(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tc001_linux.window import App

    monkeypatch.setenv("XDG_VIDEOS_DIR", str(tmp_path / "vids"))
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: app.shown > 0)
        fake_exe(tmp_path, monkeypatch, "ffmpeg", "echo 'Unknown encoder libx264' >&2; exit 1")
        app.act("record")
        assert saved(app, root, "Video failed: ") == "Unknown encoder libx264"
        pump(root, lambda: app.record_button.cget("text") == "Record")
        assert app.recorder is None
        app.videos = tmp_path / "settings.json" / "x" / "TC001"
        (tmp_path / "settings.json").touch()
        app.status.configure(text="")
        app.act("record")  # the folder cannot be made
        assert "Not a directory" in saved(app, root, "Video failed: ")
        assert app.recorder is None and app.record_button.cget("text") == "Record"
    finally:
        app.quit()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_signal_finalises_the_recording_001_AC7(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sig: signal.Signals
) -> None:
    """run() with its signal handlers: R, then SIGTERM / SIGHUP to the process; the video is saved."""
    from tc001_linux.window import run

    monkeypatch.setenv("XDG_VIDEOS_DIR", str(tmp_path))
    times: list[float] = []
    opened: list[Source] = []

    def connect(on_frame: OnFrame) -> Source:
        root = tkinter._default_root  # type: ignore[attr-defined]

        def record() -> None:
            root.focus_force()
            root.event_generate("<KeyPress>", keysym="r")
            times.append(time.monotonic())

        def term() -> None:
            times.append(time.monotonic())
            os.kill(os.getpid(), sig)

        root.after(300, record)
        root.after(2300, term)
        opened.append(ReplaySource(CAPTURE, on_frame))
        return opened[-1]

    try:
        run(connect, Settings(), tmp_path / "settings.json")
    finally:
        for src in opened:
            src.stop()
    folder = tmp_path / "TC001"
    end = time.monotonic() + 15
    while not (videos := list(folder.glob("TC001_*.mp4"))):
        assert time.monotonic() < end, "no video"
        time.sleep(0.05)
    assert list(folder.iterdir()) == videos
    video_check(videos[0], times[1] - times[0])


def test_xft_fonts_and_dark_theme_001_AC11(xvfb: None, tmp_path: Path) -> None:
    """System Python's Tk draws anti-aliased (Xft) fonts, uv's builds do not; ttk in DejaVu, dark."""
    from tkinter import font as tkfont

    from tc001_linux.window import BG, BUTTON_BG, App

    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        assert root.tk.call("::tk::pkgconfig", "get", "fontsystem") == "xft"
        style = tkinter.ttk.Style(root)
        assert style.theme_use() == "clam"
        for widget in (app.photo_button, app.lock_button, app.status, app.smoothing_value):
            assert isinstance(widget, tkinter.ttk.Widget)
            name = style.lookup(widget.winfo_class(), "font")
            assert tkfont.nametofont(name, root).actual("family") == "DejaVu Sans", widget
        assert {f.actual("family") for f in app.readout_fonts} == {"DejaVu Sans Mono"}
        assert style.lookup("TButton", "background") == BUTTON_BG and style.lookup(".", "background") == BG
        assert isinstance(app.scale, tkinter.ttk.Scale) and app.smoothing_value.cget("text") == "0.50"
    finally:
        app.quit()


def settle(app: object, root: tkinter.Tk, w: int, h: int) -> None:
    """Resizes the window and waits for the debounced layout."""
    root.geometry(f"{w}x{h}")
    pump(root, lambda: (root.winfo_width(), root.winfo_height()) == (w, h))
    pump(root, lambda: app._layout_id is None)  # type: ignore[attr-defined]
    root.update()


def box(widget: tkinter.Misc) -> tuple[int, int, int, int]:
    """x0, y0, x1, y1 in the toplevel."""
    top = widget.winfo_toplevel()
    x, y = widget.winfo_rootx() - top.winfo_rootx(), widget.winfo_rooty() - top.winfo_rooty()
    return x, y, x + widget.winfo_width(), y + widget.winfo_height()


@pytest.mark.parametrize(("w", "h"), [(600, 700), (1200, 1400), (1600, 900)])
def test_resized_window_scales_the_picture_001_AC11(xvfb: None, tmp_path: Path, w: int, h: int) -> None:
    from tc001_linux.window import READOUT_PX, App

    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        pump(root, lambda: app.readout_texts is not None)
        natural = app.photo_button.winfo_height()
        settle(app, root, w, h)
        shown = app.shown
        pump(root, lambda: app.shown > shown + 2)
        pw, ph = app.photo.width(), app.photo.height()
        assert (app.picture.winfo_width(), app.picture.winfo_height()) == (pw, ph)
        assert abs(ph * WIDTH - pw * HEIGHT) <= WIDTH  # 768:832 within 1 px of height
        x0, y0, x1, y1 = box(app.picture)
        assert 0 <= x0 and x1 <= w and 0 <= y0 and abs(x0 + x1 - w) <= 1  # inside, centred
        # Fills the room it has: limited by the width or by the height left above the readouts.
        rx0, ry0, rx1, ry1 = box(app.readouts)
        assert y1 <= ry0 and (pw == w or ry0 - y1 <= 1)
        assert abs(rx1 - rx0 - pw) <= 1  # readouts as wide as the picture
        scale = pw / WIDTH
        for f, px in zip(app.readout_fonts, READOUT_PX, strict=True):
            assert abs(-f.cget("size") - px * scale) <= 1
        assert app.centre_label.winfo_ismapped() and app.centre_label.cget("text")
        for button in (app.photo_button, app.record_button, app.palette_button, app.mirror_button):
            bx0, by0, bx1, by1 = box(button)
            assert button.winfo_ismapped() and ry1 <= by0 and by1 <= h and by1 - by0 == natural
        assert box(app.mirror_button)[2] >= w - 4 and box(app.palette_button)[0] <= 4  # full width
        assert box(app.scale)[3] <= h
        # Nearest neighbour of the composed picture.
        picture = app._shown_picture
        assert picture is not None
        for x, y in ((0, 0), (pw - 1, ph - 1), (pw // 2, ph // 3)):
            want = tuple(int(c) for c in picture[y * HEIGHT // ph, x * WIDTH // pw])
            assert tuple(app.photo.get(x, y)) == want
    finally:
        app.quit()


def test_resampled_frame_time_001_AC11(xvfb: None, tmp_path: Path) -> None:
    """The UI path (resample, PPM, put) at a large size stays well inside a 40 ms frame; --debug shows it."""
    from tc001_linux.window import App

    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json", debug=True)
    debug = app.debug
    assert debug is not None
    try:
        settle(app, root, 1200, 1400)
        pw, ph = app.photo.width(), app.photo.height()
        assert pw > WIDTH  # upscaled: the map is in use
        texts: list[str] = []

        def sampled() -> bool:
            text = str(debug.cget("text"))
            if f"at {pw}x{ph}" in text and (not texts or text != texts[-1]):
                texts.append(text)  # one per DEBUG_PERIOD
            return len(texts) >= 3

        pump(root, sampled)
        samples = sorted(float(text.split("ui ")[1].split(" ms")[0]) for text in texts)
        assert samples[1] < 35, texts  # median, loose for CI: about 15 ms on a desktop
    finally:
        app.quit()


def test_captures_stay_full_size_when_resized_001_AC11(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A photo and a recording taken while the window is resized are still 768 x 832."""
    from tc001_linux.window import App

    monkeypatch.setenv("XDG_PICTURES_DIR", str(tmp_path / "pics"))
    monkeypatch.setenv("XDG_VIDEOS_DIR", str(tmp_path / "vids"))
    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    try:
        settle(app, root, 600, 700)
        t0 = start_recording(app, root)
        rec = app.recorder
        assert rec is not None
        settle(app, root, 1200, 1400)
        app.act("photo")
        name = saved(app, root, "Photo saved: ")
        assert decode_png((tmp_path / "pics" / "TC001" / f"{name}.png").read_bytes()).shape == (
            HEIGHT,
            WIDTH,
            3,
        )
        settle(app, root, 1600, 900)
        pump(root, lambda: time.monotonic() - t0 >= 2.0)
        app.act("record")
        wall = time.monotonic() - t0
        saved(app, root, "Video saved: ")
        video_check(rec.path, wall)
    finally:
        app.quit()


def test_quit_leaves_no_pending_tick(xvfb: None, tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    from tc001_linux.window import App

    root = tkinter.Tk()
    App(root, replay, Settings(), tmp_path / "settings.json").quit()
    other = tkinter.Tk()  # its event loop would run the destroyed window's tick
    try:
        end = time.monotonic() + 0.3
        while time.monotonic() < end:
            other.update()
            time.sleep(0.01)
    finally:
        other.destroy()
    assert "invalid command name" not in capfd.readouterr().err


def test_photo_without_a_frame_says_so(xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No source: refused at once; a source that sends nothing: a wait line, then a failure on loss."""
    from tc001_linux import window
    from tc001_linux.window import App

    monkeypatch.setattr(window, "PHOTO_WAIT", 0.2)

    class Silent(Source):
        def __init__(self, on_frame: OnFrame) -> None:
            super().__init__(on_frame)
            self._thread.start()

        def _run(self) -> None:
            self._stop.wait()

    sources: list[Source] = []

    def connect(on_frame: OnFrame) -> Source | None:
        sources.append(Silent(on_frame))
        return sources[-1]

    root = tkinter.Tk()
    app = App(root, connect, Settings(), tmp_path / "settings.json")
    try:
        app.act("photo")  # before the first poll: no source yet
        assert saved(app, root, "Photo failed: ", timeout=0.5) == "no frame"  # at once, not on a later loss
        pump(root, lambda: app.source is not None)
        app.act("photo")
        assert saved(app, root, "Photo: ") == "waiting for a frame"
        app.status.configure(text="")
        pump(root, lambda: app.source is None)  # stalled 3 s: lost
        assert saved(app, root, "Photo failed: ") == "no frame"
        assert not list(tmp_path.glob("**/*.png"))
    finally:
        app.quit()


def test_dead_stderr_keeps_the_status_001_AC6(
    xvfb: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stderr gone (terminal closed): the photo is saved and the status line says so, once."""
    from tc001_linux.window import App

    class Dead:
        def write(self, text: str) -> int:
            raise OSError(5, "Input/output error")

    root = tkinter.Tk()
    app = App(root, replay, Settings(), tmp_path / "settings.json")
    folder = tmp_path / "Pictures" / "TC001"  # conftest media_dirs
    try:
        pump(root, lambda: app.shown > 0)
        monkeypatch.setattr(sys, "stderr", Dead())
        app.act("photo")
        name = saved(app, root, "Photo saved: ")
        assert sorted(p.name for p in folder.iterdir()) == [f"{name}.bin", f"{name}.png"]
        assert app._statuses.empty()  # no second, "failed" message behind it
    finally:
        app.quit()
