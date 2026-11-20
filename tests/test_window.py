"""Window smoke test on a private Xvfb; skipped without Xvfb locally, failed without it in CI (001-AC9)."""

import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tkinter
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tc001_linux import settings as cfg
from tc001_linux.frame import FRAME, parse
from tc001_linux.picture import BAR_MARGIN, BAR_TOP, BAR_WIDTH, WIDTH, bar_row, format_celsius, readout_panel
from tc001_linux.settings import Settings
from tc001_linux.source import CameraSource, OnFrame, ReplaySource, Source

from .conftest import CAPTURE, fake_v4l2


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
def xvfb(x_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", x_server)


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

    from tc001_linux.window import App, ppm

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
        assert pixels(app.panel_photo.data(format="ppm")) == readout_panel(r).tobytes()
        assert pixels(app.photo.data(format="ppm")) == pixels(shot.picture)

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
        assert len(ppm(readout_panel(None))) > 0
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
    """Motions and release queued together: -command runs at idle, after the release handler."""
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
        assert app.lock_button.cget("text") == "Fixed range" and app.lock_button.cget("state") == "disabled"
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

    def counting(r: object) -> object:
        if r is not None:
            refreshed.append(time.monotonic())
        return readout_panel(r)  # type: ignore[arg-type]

    monkeypatch.setattr(window, "readout_panel", counting)
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
        assert [sig for sig, _ in calls[:2]] == [signal.SIGINT, signal.SIGTERM]
        assert calls[2:] == [(sig, f"previous {sig}") for sig in (signal.SIGINT, signal.SIGTERM)]  # restored
        assert len(opened) == 1 and not opened[0].alive
    finally:
        for src in opened:
            src.stop()
        if tkinter._default_root is not None:  # type: ignore[attr-defined]
            tkinter._default_root.destroy()  # type: ignore[attr-defined]


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
