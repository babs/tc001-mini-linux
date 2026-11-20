"""Viewer settings kept across runs, the keyboard map and the actions that change the settings."""

import dataclasses
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tc001_linux.render import PALETTES

ROTATIONS = (0, 90, 180, 270)
SMOOTH_STEP = 0.1

# Lower-cased Tk keysyms.
KEYS = {
    "p": "photo",
    "r": "record",
    "c": "palette",
    "l": "lock",
    "t": "rotate",
    "m": "mirror",
    "plus": "smooth+",
    "kp_add": "smooth+",
    "minus": "smooth-",
    "kp_subtract": "smooth-",
    "q": "quit",
}


@dataclass(frozen=True, slots=True)
class Settings:
    """Defaults as the Android app's Settings."""

    palette: str = "ironbow"
    rotation: int = 0
    mirror: bool = False
    smoothing: float = 0.5


def key_action(keysym: str) -> str | None:
    return KEYS.get(keysym.lower())


def apply(s: Settings, action: str) -> Settings:
    """`s` after a settings action (palette, rotate, mirror, smooth+, smooth-); other actions leave it."""
    match action:
        case "palette":
            names = list(PALETTES)
            return dataclasses.replace(s, palette=names[(names.index(s.palette) + 1) % len(names)])
        case "rotate":
            return dataclasses.replace(s, rotation=(s.rotation + 90) % 360)
        case "mirror":
            return dataclasses.replace(s, mirror=not s.mirror)
        case "smooth+" | "smooth-":
            step = SMOOTH_STEP if action == "smooth+" else -SMOOTH_STEP
            return dataclasses.replace(s, smoothing=min(max(round(s.smoothing + step, 2), 0.0), 1.0))
    return s


def _valid(name: str, v: object) -> bool:
    match name:
        case "palette":
            return isinstance(v, str) and v in PALETTES
        case "rotation":
            return type(v) is int and v in ROTATIONS
        case "mirror":
            return isinstance(v, bool)
        case "smoothing":
            num = isinstance(v, int | float) and not isinstance(v, bool)
            return num and math.isfinite(v) and 0 <= v <= 1  # type: ignore[arg-type,operator]
    return False


def parse(text: str) -> Settings:
    """Settings from JSON; a missing or invalid value falls back to its default, unknown keys are ignored."""
    try:
        data = json.loads(text)
    except ValueError:
        return Settings()
    if not isinstance(data, dict):
        return Settings()
    good = {f.name: data[f.name] for f in dataclasses.fields(Settings) if _valid(f.name, data.get(f.name))}
    if "smoothing" in good:
        good["smoothing"] = float(good["smoothing"])
    return Settings(**good)


def dump(s: Settings) -> str:
    return json.dumps(dataclasses.asdict(s), indent=2) + "\n"


def path(env: Mapping[str, str] = os.environ) -> Path:
    """$XDG_CONFIG_HOME/tc001-mini-linux/settings.json; a relative or empty XDG_CONFIG_HOME is ignored."""
    xdg = env.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if os.path.isabs(xdg) else Path(env.get("HOME", Path.home())) / ".config"
    return base / "tc001-mini-linux" / "settings.json"


def load(p: Path) -> Settings:
    try:
        return parse(p.read_text())
    except OSError, UnicodeDecodeError:
        return Settings()


def save(p: Path, s: Settings) -> None:
    """Atomic write: a crash mid-write leaves the old file, never a truncated one."""
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".settings.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(dump(s))
            f.flush()
            os.fsync(f.fileno())  # data on disk before the rename, or a power cut can leave an empty file
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
