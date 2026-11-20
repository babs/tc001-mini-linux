import io
import os
import struct
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tc001_linux import Image, text
from tc001_linux.cli import SNAPSHOT_FRAME, main, snapshot
from tc001_linux.frame import FRAME, parse
from tc001_linux.picture import HEIGHT, WIDTH, compose
from tc001_linux.process import Processor
from tc001_linux.settings import Settings, save
from tc001_linux.source import CameraSource, ReplaySource

from .conftest import CAPTURE, decode_png


def composed(
    capture: bytes,
    palette: str,
    rotate: int,
    mirror: bool,
    smooth: float,
    fixed: tuple[float, float] | None = None,
) -> Image:
    """Frame 5 through Processor + compose, written out by hand."""
    proc = Processor(fixed)
    for i in range(SNAPSHOT_FRAME):
        t = parse(capture[i * FRAME : (i + 1) * FRAME])
        assert t is not None
        shown, r = proc.process(t, smooth)
    return compose(shown, r, palette, smooth, rotate, mirror)


@pytest.mark.parametrize(
    ("args", "palette", "rotate", "mirror", "smooth"),
    [
        ([], "ironbow", 0, False, 0.5),
        (
            ["--palette", "black-hot", "--rotate", "270", "--mirror", "--smooth", "0"],
            "black-hot",
            270,
            True,
            0.0,
        ),
    ],
)
def test_snapshot_is_frame5_composed_001_AC3(
    tmp_path: Path, capture: bytes, args: list[str], palette: str, rotate: int, mirror: bool, smooth: float
) -> None:
    out = tmp_path / "snap.png"
    main(["--file", str(CAPTURE), "--snapshot", str(out), *args])
    img = decode_png(out.read_bytes())
    assert img.shape == (HEIGHT, WIDTH, 3)
    assert np.array_equal(img, composed(capture, palette, rotate, mirror, smooth))
    assert not np.array_equal(img, composed(capture, palette, (rotate + 90) % 360, mirror, smooth))


def test_snapshot_fixed_range(capture: bytes, tmp_path: Path) -> None:
    out = tmp_path / "snap.png"
    main(["--file", str(CAPTURE), "--snapshot", str(out), "--range", "0", "100"])
    img = decode_png(out.read_bytes())
    assert np.array_equal(img, composed(capture, "ironbow", 0, False, 0.5, (0.0, 100.0)))
    assert not np.array_equal(img, composed(capture, "ironbow", 0, False, 0.5))


def test_snapshot_short_source(capture: bytes, tmp_path: Path) -> None:
    assert snapshot(io.BytesIO(capture[: 4 * FRAME])) is None
    short = tmp_path / "short.bin"
    short.write_bytes(capture[: 4 * FRAME])
    with pytest.raises(SystemExit, match="ends before frame 5"):
        main(["--file", str(short), "--snapshot", str(tmp_path / "x.png")])


def test_headless_subprocess_001_AC9(tmp_path: Path) -> None:
    """The console entry point renders with no display and never imports Tk."""
    out = tmp_path / "snap.png"
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    code = (
        "import sys; from tc001_linux.cli import main; "
        f"main(['--file', {str(CAPTURE)!r}, '--snapshot', {str(out)!r}]); "
        "import tc001_linux.source, tc001_linux.settings; "
        "assert 'tkinter' not in sys.modules and 'tc001_linux.window' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=60)
    assert decode_png(out.read_bytes()).shape == (HEIGHT, WIDTH, 3)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--snapshot", "x.png"], "add --file CAPTURE"),
        (["--file", "/nonexistent.bin", "--snapshot", "x.png"], "No such file"),
        (["--file", "/nonexistent.bin"], "No such file"),
        (["--file", __file__], "no mode-2 frame in"),
    ],
)
def test_exits_with_a_message(argv: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        main(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--smooth", "1.5"],
        ["--smooth", "nan"],
        ["--range", "0", "inf"],
        ["--rotate", "45"],
        ["--palette", "rainbow"],
        ["--range", "30", "20"],
        ["--range", "0", "1e5"],
        ["--range", "-1e39", "0"],
        ["--range", "20", "20"],
        ["--device", "/dev/video0"],
    ],
)
def test_bad_arguments(argv: list[str], capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out = tmp_path / "x.png"
    with pytest.raises(SystemExit) as e:
        main(["--file", str(CAPTURE), "--snapshot", str(out), *argv])
    assert not out.exists()
    assert e.value.code == 2 and "error:" in capsys.readouterr().err


def test_font_error_exits_readably(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh_font: None) -> None:
    monkeypatch.setattr(text, "FONT_FILE", tmp_path / "missing.ttf")
    with pytest.raises(SystemExit, match="tc001-mini-linux: font not found"):
        main(["--file", str(CAPTURE), "--snapshot", str(tmp_path / "x.png")])
    assert not (tmp_path / "x.png").exists()


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A stand-in for tc001_linux.window that records what run() gets; the settings live in tmp_path."""
    got: dict[str, Any] = {}
    fake = types.ModuleType("tc001_linux.window")
    fake.run = lambda connect, settings, path, fixed, debug: got.update(  # type: ignore[attr-defined]
        connect=connect, settings=settings, path=path, fixed=fixed, debug=debug
    )
    monkeypatch.setitem(sys.modules, "tc001_linux.window", fake)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return got


def test_window_start_sweeps_the_media_folders_001_AC7(
    window: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    videos, pictures = tmp_path / "Videos" / "TC001", tmp_path / "Pictures" / "TC001"  # conftest media_dirs
    videos.mkdir(parents=True)
    pictures.mkdir(parents=True)
    moov = struct.pack(">I", 8) + b"moov"
    (videos / ".TC001_x.mp4.part").write_bytes(moov)
    (pictures / ".TC001_y.png.part").write_bytes(b"png")
    main(["--file", str(CAPTURE)])
    assert list(videos.iterdir()) == [videos / "TC001_x.mp4"] and list(pictures.iterdir()) == []
    err = capsys.readouterr().err
    assert (
        "tc001-mini-linux: recovered TC001_x.mp4" in err
        and "tc001-mini-linux: discarded .TC001_y.png.part" in err
    )


def test_window_gets_saved_settings_and_overrides_001_AC5(window: dict[str, Any], tmp_path: Path) -> None:
    saved = tmp_path / "cfg" / "tc001-mini-linux" / "settings.json"
    save(saved, Settings("black-hot", 90, False, 0.4))
    main(["--file", str(CAPTURE), "--debug", "--range", "10", "30"])
    assert window["settings"] == Settings("black-hot", 90, False, 0.4)
    assert (window["path"], window["fixed"], window["debug"]) == (saved, [10.0, 30.0], True)
    main(["--file", str(CAPTURE), "--palette", "ironbow", "--mirror", "--smooth", "0", "--rotate", "0"])
    assert window["settings"] == Settings("ironbow", 0, True, 0.0)
    save(saved, Settings(mirror=True))
    main(["--file", str(CAPTURE), "--no-mirror"])
    assert window["settings"] == Settings(mirror=False)
    main(["--file", str(CAPTURE)])
    assert window["settings"] == Settings(mirror=True)  # neither flag: the saved value


def test_window_connects_to_the_file(window: dict[str, Any], tmp_path: Path) -> None:
    copy = tmp_path / "c.bin"
    copy.write_bytes(CAPTURE.read_bytes())
    main(["--file", str(copy)])
    src = window["connect"](lambda raw, t: None)
    assert isinstance(src, ReplaySource)
    src.stop()
    copy.unlink()
    assert window["connect"](lambda raw, t: None) is None


def test_window_connects_to_the_camera(
    window: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "bin" / "v4l2-ctl"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\nexec sleep 30\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")
    found: list[str | None] = [None]
    monkeypatch.setattr("tc001_linux.cli.find_device", lambda: found[0])
    main([])
    assert window["connect"](lambda raw, t: None) is None  # no camera plugged in
    node = tmp_path / "video5"
    node.touch()
    found[0] = str(node)
    src = window["connect"](lambda raw, t: None)
    assert isinstance(src, CameraSource) and src.device == str(node)
    src.stop()
    main(["--device", str(tmp_path / "video9")])
    assert window["connect"](lambda raw, t: None) is None  # --device not there (yet)


def test_window_needs_v4l2_ctl(window: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(SystemExit, match="v4l2-ctl not found"):
        main([])
    assert not window
