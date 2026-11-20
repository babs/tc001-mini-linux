import json
import os
from pathlib import Path

import pytest

from tc001_linux import settings as cfg
from tc001_linux.settings import Settings


def test_defaults_match_android() -> None:
    """tc001-mini-android core/Settings.kt: IRONBOW, 0 quarter turns, no mirror, smoothing 0.5f."""
    assert Settings() == Settings(palette="ironbow", rotation=0, mirror=False, smoothing=0.5)


def test_round_trip_001_AC5(tmp_path: Path) -> None:
    s = Settings(palette="black-hot", rotation=270, mirror=True, smoothing=0.35)
    assert cfg.parse(cfg.dump(s)) == s
    p = tmp_path / "deep" / "settings.json"
    cfg.save(p, s)
    assert cfg.load(p) == s
    assert os.listdir(p.parent) == ["settings.json"]  # no temporary file left behind


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("", Settings()),
        ("not json", Settings()),
        ("[1, 2]", Settings()),
        ('"ironbow"', Settings()),
        ('{"palette": "rainbow", "rotation": 90}', Settings(rotation=90)),
        ('{"rotation": 45, "mirror": true}', Settings(mirror=True)),
        ('{"rotation": 90.0}', Settings()),
        ('{"rotation": true}', Settings()),
        ('{"mirror": 1, "smoothing": 0.25}', Settings(smoothing=0.25)),
        ('{"smoothing": 0}', Settings(smoothing=0.0)),
        ('{"smoothing": 1.5}', Settings()),
        ('{"smoothing": -0.1}', Settings()),
        ('{"smoothing": NaN}', Settings()),
        ('{"smoothing": Infinity}', Settings()),
        ('{"smoothing": "0.5"}', Settings()),
        ('{"smoothing": true}', Settings()),
        ('{"smoothing": 1, "palette": "white-hot", "extra": 3}', Settings("white-hot", smoothing=1.0)),
    ],
)
def test_invalid_values_fall_back_001_AC5(text: str, want: Settings) -> None:
    got = cfg.parse(text)
    assert got == want
    assert type(got.smoothing) is float


def test_load_missing_or_unreadable(tmp_path: Path) -> None:
    assert cfg.load(tmp_path / "none.json") == Settings()
    (tmp_path / "bin.json").write_bytes(b"\xff\xfe\x00")
    assert cfg.load(tmp_path / "bin.json") == Settings()


def test_save_replaces_and_keeps_old_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "settings.json"
    cfg.save(p, Settings(palette="white-hot"))
    cfg.save(p, Settings(palette="black-hot"))
    assert json.loads(p.read_text())["palette"] == "black-hot"

    def boom(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        cfg.save(p, Settings(palette="ironbow"))
    assert json.loads(p.read_text())["palette"] == "black-hot"
    assert os.listdir(tmp_path) == ["settings.json"]


@pytest.mark.parametrize(
    ("env", "want"),
    [
        ({"XDG_CONFIG_HOME": "/x/cfg", "HOME": "/h"}, "/x/cfg/tc001-mini-linux/settings.json"),
        ({"XDG_CONFIG_HOME": "", "HOME": "/h"}, "/h/.config/tc001-mini-linux/settings.json"),
        ({"XDG_CONFIG_HOME": "rel/cfg", "HOME": "/h"}, "/h/.config/tc001-mini-linux/settings.json"),
        ({"HOME": "/h"}, "/h/.config/tc001-mini-linux/settings.json"),
    ],
)
def test_path_xdg(env: dict[str, str], want: str) -> None:
    assert cfg.path(env) == Path(want)


@pytest.mark.parametrize(
    ("keysym", "action"),
    [
        ("p", "photo"),
        ("P", "photo"),
        ("r", "record"),
        ("c", "palette"),
        ("C", "palette"),
        ("l", "lock"),
        ("t", "rotate"),
        ("m", "mirror"),
        ("plus", "smooth+"),
        ("KP_Add", "smooth+"),
        ("minus", "smooth-"),
        ("KP_Subtract", "smooth-"),
        ("q", "quit"),
        ("Q", "quit"),
        ("x", None),
        ("Return", None),
        ("", None),
    ],
)
def test_key_action_001_AC5(keysym: str, action: str | None) -> None:
    assert cfg.key_action(keysym) == action


def test_apply_actions_001_AC5() -> None:
    s = Settings()
    palettes = []
    for _ in range(3):
        s = cfg.apply(s, "palette")
        palettes.append(s.palette)
    assert palettes == ["white-hot", "black-hot", "ironbow"]
    rotations = [s.rotation]
    for _ in range(4):
        s = cfg.apply(s, "rotate")
        rotations.append(s.rotation)
    assert rotations == [0, 90, 180, 270, 0]
    assert cfg.apply(s, "mirror").mirror and not cfg.apply(cfg.apply(s, "mirror"), "mirror").mirror
    for _ in range(12):
        s = cfg.apply(s, "smooth+")
    assert s.smoothing == 1.0
    s = cfg.apply(cfg.apply(s, "smooth-"), "smooth-")
    assert s.smoothing == 0.8
    for _ in range(12):
        s = cfg.apply(s, "smooth-")
    assert s.smoothing == 0.0
    assert cfg.apply(Settings(), "smooth+").smoothing == 0.6
    assert cfg.apply(Settings(), "smooth-").smoothing == 0.4
    for other in ("photo", "record", "lock", "quit", ""):
        assert cfg.apply(s, other) is s


def test_save_syncs_before_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fsync, replace = os.fsync, os.replace

    def logged_fsync(fd: int) -> None:
        calls.append("fsync")
        fsync(fd)

    def logged_replace(src: str, dst: str) -> None:
        calls.append("replace")
        replace(src, dst)

    monkeypatch.setattr(os, "fsync", logged_fsync)
    monkeypatch.setattr(os, "replace", logged_replace)
    cfg.save(tmp_path / "settings.json", Settings())
    assert calls == ["fsync", "replace"]
