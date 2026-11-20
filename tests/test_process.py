import numpy as np
import pytest

from tc001_linux import Plane
from tc001_linux.frame import FRAME, SIZE, parse
from tc001_linux.process import (
    Extremes,
    Processor,
    Readouts,
    Spot,
    TemporalSmoother,
    box3,
    centre_celsius,
    clamp_strength,
    span_hi,
)


def flat(v: float = 20.0) -> Plane:
    return np.full((SIZE, SIZE), v, np.float32)


def test_box3_edges_padded() -> None:
    t = flat(0.0)
    t[0, 0] = 9.0
    s = box3(t)
    assert s[0, 0] == pytest.approx(4.0) and s[1, 1] == pytest.approx(1.0) and s[2, 2] == 0.0


def test_tracker_hysteresis() -> None:
    scene = flat()
    scene[10:13, 10:13] = 30.0
    tr = Extremes(hyst=0.3)
    assert tr.update(scene)[0][:2] == (11, 11)
    scene[50:53, 50:53] = 30.2  # beats the old spot by less than hyst: spot stays put
    assert tr.update(scene)[0][:2] == (11, 11)
    scene[50:53, 50:53] = 31.0  # clear winner: spot moves
    assert tr.update(scene)[0][:2] == (51, 51)
    scene[0:3, 0:3] = -5.0
    assert tr.update(scene)[1][:2] == (0, 0)


def test_tracker_label_ema() -> None:
    scene = flat()
    scene[40:43, 40:43] = 30.0
    tr = Extremes(hyst=0.3, alpha=0.3)
    assert tr.update(scene)[0].celsius == pytest.approx(30.0)
    scene[40:43, 40:43] = 30.2  # same place, below hysteresis: label eases 30% of the way
    assert tr.update(scene)[0].celsius == pytest.approx(30.06, abs=1e-4)


def test_clamp_strength() -> None:
    assert [clamp_strength(s) for s in (-1.0, 0.0, 0.5, 2.0, float("nan"))] == [0.0, 0.0, 0.5, 1.0, 0.0]


def test_temporal_smoothing() -> None:
    sm = TemporalSmoother()
    a, b = flat(10.0), flat(20.0)
    assert sm.apply(a, 0.0) is a  # strength 0: the frame itself
    first = sm.apply(a, 0.5)
    assert first is not a and float(first[0, 0]) == 10.0
    assert float(sm.apply(b, 0.5)[0, 0]) == pytest.approx(17.0)  # alpha 0.7 at 0.5
    assert float(sm.apply(b, 0.0)[0, 0]) == 20.0  # back to 0 drops the history
    sm.apply(a, 1.0)
    assert float(sm.apply(b, 1.0)[0, 0]) == pytest.approx(14.0)  # alpha 0.4 at 1.0


def test_span_and_centre() -> None:
    assert span_hi(25.0, 25.0) == pytest.approx(25.1) and span_hi(20.0, 30.0) == 30.0
    t = flat()
    t[47:49, 47:49] = [[30.0, 31.0], [32.0, 33.0]]
    assert centre_celsius(t) == 31.5


def test_readouts_values_001_AC3() -> None:
    """Centre = raw 2x2 mean; Min/Max = the marker values, not the raw extremes."""
    t = flat()
    t[47:49, 47:49] = [[30.0, 31.0], [32.0, 33.0]]
    t[0, 0] = 99.0  # edge padding: the 3x3 box at the corner holds it 4 times
    t[90:93, 10:13] = 5.0
    shown, r = Processor().process(t, 0.0)
    assert shown is t
    assert r.centre == 31.5
    assert (
        (r.hot.row, r.hot.col) == (0, 0)
        and r.max == pytest.approx((4 * 99 + 5 * 20) / 9)
        and r.max != float(t.max())
    )
    assert (r.cold.row, r.cold.col) == (91, 11) and r.min == pytest.approx(5.0)
    assert (r.lo, r.hi, r.mid) == (5.0, 99.0, 52.0) and not r.locked


def test_readouts_properties() -> None:
    r = Readouts(Spot(1, 2, 40.5), Spot(3, 4, -3.25), 31.5, 20.0, 30.0, False)
    assert (r.min, r.max, r.mid) == (-3.25, 40.5, 25.0)


def test_flat_scene_no_division_by_zero() -> None:
    _, r = Processor().process(flat(25.0), 0.0)
    assert r.hi - r.lo >= 0.1 - 1e-6


def test_locked_range() -> None:
    p = Processor()
    a, b = flat(20.0), flat(20.0)
    a[0, 0], b[0, 0] = 30.0, 40.0
    _, r = p.process(a, 0.0)
    assert (r.lo, r.hi, r.locked) == (20.0, 30.0, False)
    p.locked = True
    _, r = p.process(b, 0.0)
    assert (r.lo, r.hi, r.locked) == (20.0, 30.0, True)  # frozen at the last unlocked frame
    p.locked = False
    _, r = p.process(b, 0.0)
    assert (r.lo, r.hi) == (20.0, 40.0)


def test_locked_before_first_frame_takes_its_range() -> None:
    p = Processor()
    p.locked = True
    t = flat(20.0)
    t[0, 0] = 30.0
    _, r = p.process(t, 0.0)
    assert (r.lo, r.hi, r.locked) == (20.0, 30.0, True)


def test_fixed_range() -> None:
    _, r = Processor(fixed=(20.0, 30.0)).process(flat(25.0), 0.0)
    assert (r.lo, r.hi, r.locked) == (20.0, 30.0, True)
    _, r = Processor(fixed=(20.0, 20.0)).process(flat(25.0), 0.0)
    assert r.hi == pytest.approx(20.1)


def test_range_from_raw_not_smoothed(capture: bytes) -> None:
    p = Processor()
    for i in range(5):
        parsed = parse(capture[i * FRAME : (i + 1) * FRAME])
        assert parsed is not None
        t = parsed
        shown, r = p.process(t, 1.0)
    assert (r.lo, r.hi) == (float(t.min()), float(t.max()))
    assert not np.array_equal(shown, t)
