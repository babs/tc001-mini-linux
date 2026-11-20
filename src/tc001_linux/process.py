"""Per-frame state: spot tracker, colour range, centre readout and display-only temporal smoothing.

Arithmetic stays in float32 as in the Android app, so labels round to the same tenth.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from tc001_linux import Plane
from tc001_linux.frame import SIZE


class Spot(NamedTuple):
    """Raw-plane position and label value of a tracked spot."""

    row: int
    col: int
    celsius: float


def box3(t: Plane) -> Plane:
    """3x3 box mean, edges padded."""
    p = np.pad(t, 1, mode="edge")
    acc = np.zeros_like(t)
    for dy in range(3):
        for dx in range(3):
            acc += p[dy : dy + t.shape[0], dx : dx + t.shape[1]]
    return acc / np.float32(9)


class Extremes:
    """Hot/cold spot tracker: a spot only moves when a new place beats it by `hyst` C (noise-proof)."""

    def __init__(self, hyst: float = 0.3, alpha: float = 0.3) -> None:
        self.hyst, self.alpha = hyst, alpha
        self._pos: dict[int, tuple[int, int]] = {}
        self._val: dict[int, np.float32] = {}

    def update(self, t: Plane) -> tuple[Spot, Spot]:
        """(hot, cold) spots after frame `t`."""
        s = box3(t)
        out = []
        for sign in (1, -1):
            i = int(np.argmax(sign * s))
            cand = (i // s.shape[1], i % s.shape[1])
            cur = self._pos.get(sign)
            if cur is None or sign * (s[cand] - s[cur]) > self.hyst:
                self._pos[sign], self._val[sign] = cand, s[cand]
            else:
                v = self._val[sign]
                self._val[sign] = v + np.float32(self.alpha) * (s[cur] - v)  # EMA steadies the label
            out.append(Spot(*self._pos[sign], float(self._val[sign])))
        return out[0], out[1]


def clamp_strength(s: float) -> float:
    """Smoothing strength in [0, 1]; NaN gives 0 (it would extrapolate past the LUT)."""
    return min(s, 1.0) if s > 0 else 0.0


class TemporalSmoother:
    """Display-only EMA, alpha = 1 - 0.6 s; s = 0 returns the frame itself and drops the history."""

    def __init__(self) -> None:
        self._shown: Plane | None = None

    def apply(self, t: Plane, strength: float) -> Plane:
        s = clamp_strength(strength)
        if s == 0:
            self._shown = None
            return t
        if self._shown is None:
            self._shown = t.copy()
        else:
            self._shown += (np.float32(1) - np.float32(0.6) * np.float32(s)) * (t - self._shown)
        return self._shown


def span_hi(lo: float, hi: float) -> float:
    """`hi` raised so lo..hi is never narrower than 0.1 C (flat scene)."""
    return max(hi, float(np.float32(lo) + np.float32(0.1)))


def centre_celsius(t: Plane) -> float:
    """Mean of the 2x2 centre pixels: the same four pixels whatever the rotation or mirror."""
    c = SIZE // 2
    return float((t[c - 1, c - 1] + t[c - 1, c] + t[c, c - 1] + t[c, c]) / np.float32(4))


@dataclass(frozen=True, slots=True)
class Readouts:
    """The numbers of one frame: markers, centre badge, scale labels (and the window's Min / Max / Center)."""

    hot: Spot
    cold: Spot
    centre: float
    lo: float
    hi: float
    locked: bool

    @property
    def min(self) -> float:
        return self.cold.celsius

    @property
    def max(self) -> float:
        return self.hot.celsius

    @property
    def mid(self) -> float:
        return float((np.float32(self.lo) + np.float32(self.hi)) / np.float32(2))


class Processor:
    """Spots, centre and colour range from the raw frame; only the returned plane is smoothed.

    `locked` keeps lo..hi at their values of the last unlocked frame; `fixed` forces lo..hi.
    """

    def __init__(self, fixed: tuple[float, float] | None = None) -> None:
        self.fixed = fixed
        self.locked = False
        self.spots = Extremes()
        self._smoother = TemporalSmoother()
        self._range: tuple[float, float] | None = None

    def process(self, t: Plane, strength: float) -> tuple[Plane, Readouts]:
        hot, cold = self.spots.update(t)
        if self.fixed is not None:
            lo, hi = self.fixed
            self._range = (lo, span_hi(lo, hi))
        elif not self.locked or self._range is None:
            lo = float(t.min())
            self._range = (lo, span_hi(lo, float(t.max())))
        held = self.locked or self.fixed is not None
        r = Readouts(hot, cold, centre_celsius(t), *self._range, held)
        return self._smoother.apply(t, strength), r
