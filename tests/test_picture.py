from pathlib import Path

import numpy as np
import pytest

from tc001_linux import Image, Plane, text
from tc001_linux.frame import FRAME, SIZE, parse
from tc001_linux.picture import (
    BAR_HEIGHT,
    BAR_MARGIN,
    BAR_TOP,
    BAR_WIDTH,
    CENTRE,
    CENTRE_SIZE,
    COLD,
    FONT_SIZE,
    HALF,
    HEIGHT,
    HOT,
    LABEL_SIZE,
    OUTLINE,
    PAIR_SIZE,
    PANEL,
    SCALE,
    SIDE,
    STROKE,
    WIDTH,
    badge_width,
    compose,
    format_celsius,
    marker_label_x,
    readout_panel,
    readout_texts,
    scale_labels,
)
from tc001_linux.png import encode_png
from tc001_linux.process import Processor, Readouts, Spot
from tc001_linux.render import PALETTES, orient, orient_pos, thermal, upscale
from tc001_linux.text import FontError, draw_text, font, text_width

from .conftest import decode_png

GREY = 127  # white-hot LUT at norm 0.5
DARK = GREY // 3  # outline over GREY
C = SIDE // 2


def flat(v: float = 20.0) -> Plane:
    return np.full((SIZE, SIZE), v, np.float32)


def readouts(hot: tuple[int, int] = (10, 20), cold: tuple[int, int] = (60, 70)) -> Readouts:
    return Readouts(Spot(*hot, 30.0), Spot(*cold, 10.0), 20.0, 0.0, 40.0, False)


def grey_picture(r: Readouts, rotate: int = 0, mirror: bool = False) -> Image:
    """Picture of a flat mid-range scene in white-hot: every overlay pixel stands out from GREY."""
    return compose(flat(20.0), r, "white-hot", 0.0, rotate, mirror)


def frame(capture: bytes, i: int) -> Plane:
    t = parse(capture[i * FRAME : (i + 1) * FRAME])
    assert t is not None
    return t


# Palettes and image -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "cold", "hot"),
    [
        ("ironbow", [0, 0, 0], [255] * 3),
        ("white-hot", [0, 0, 0], [255] * 3),
        ("black-hot", [255] * 3, [0] * 3),
    ],
)
def test_palette_ends(name: str, cold: list[int], hot: list[int]) -> None:
    assert PALETTES[name][0].tolist() == cold and PALETTES[name][-1].tolist() == hot
    assert PALETTES[name].shape == (256, 3)


def test_palette_truncates_like_android() -> None:
    # 0x50008C at stop 0.25: index 64 is x = 0.251, past the stop; index 63 = 63/255 of the way, truncated.
    assert PALETTES["ironbow"][63].tolist() == [int(80 * 63 / 63.75), 0, int(140 * 63 / 63.75)]


def test_upscale_linear() -> None:
    a = np.array([[0.0, 1.0], [2.0, 3.0]], np.float32)
    up = upscale(a, 4, 1.0)
    assert up.shape == (8, 8) and up[0, 0] == 0.0 and up[-1, -1] == 3.0  # corners clamp to source
    assert np.all(np.diff(up, axis=1) >= 0) and np.all(np.diff(up, axis=0) >= 0)  # monotone ramp
    assert np.array_equal(upscale(a, 4, 0.0), a.repeat(4, 0).repeat(4, 1))


def test_smoothing_strength_ordering(capture: bytes) -> None:
    t = frame(capture, 0)
    lo, hi = float(t.min()), float(t.max())
    sharp, half, full = (thermal(t, lo, hi, "ironbow", s, 0, False, 6).astype(int) for s in (0.0, 0.5, 1.0))
    d_half, d_full = np.abs(half - sharp).sum(), np.abs(full - sharp).sum()
    assert 0 < d_half < d_full  # strength 0.5 sits between sharp and full


ORIENTED = {
    (0, False): (2, 5),
    (90, False): (5, 93),
    (180, False): (93, 90),
    (270, False): (90, 2),
    (0, True): (2, 90),
    (90, True): (5, 2),
    (180, True): (93, 5),
    (270, True): (90, 93),
}


@pytest.mark.parametrize(("rot", "mir"), ORIENTED)
def test_orientation_mapping_001_AC3(rot: int, mir: bool) -> None:
    probe = np.zeros((SIZE, SIZE), np.float32)
    probe[2, 5] = 1.0
    want = ORIENTED[rot, mir]
    assert orient_pos(2, 5, rot, mir) == want
    got = np.unravel_index(int(np.argmax(orient(probe, rot, mir))), (SIZE, SIZE))
    assert (int(got[0]), int(got[1])) == want


@pytest.mark.parametrize(("rot", "mir"), ORIENTED)
def test_marker_follows_its_raw_pixel_001_AC3(rot: int, mir: bool) -> None:
    """The hot marker frames the hot raw pixel wherever orientation moves it."""
    t = flat(20.0)
    t[1:4, 4:7] = 40.0  # 3x3 block: the tracker peaks on its centre, raw pixel (2, 5)
    t[50:53, 50:53] = 10.0  # cold spot (and its badge) away from the probe
    proc = Processor()
    shown, r = proc.process(t, 0.0)
    pic = compose(shown, r, "white-hot", 0.0, rot, mir)
    row, col = ORIENTED[rot, mir]
    y, x = row * SCALE + SCALE // 2, col * SCALE + SCALE // 2
    assert pic[y, x].tolist() == [255] * 3  # the raw pixel, white-hot
    side = x - HALF if x - HALF >= STROKE else x + HALF - 1  # left side, unless clipped by the edge
    assert pic[y, side].tolist() == list(HOT)


# Picture layout ---------------------------------------------------------------------------------


@pytest.mark.parametrize("palette", list(PALETTES))
@pytest.mark.parametrize("values", [(20.0, 20.1), (-40.0, 400.0), (-999.9, 999.9)])
def test_picture_size_001_AC3(palette: str, values: tuple[float, float]) -> None:
    lo, hi = values
    r = Readouts(Spot(0, 0, hi), Spot(95, 95, lo), lo, lo, hi, False)
    pic = compose(flat(lo), r, palette, 0.5, 90, True)
    assert pic.shape == (HEIGHT, WIDTH, 3) == (832, 768, 3) and pic.dtype == np.uint8


@pytest.mark.parametrize("palette", list(PALETTES))
def test_scale_bar_cold_left_hot_right_001_AC3(palette: str) -> None:
    pic = compose(flat(), readouts(), palette, 0.0, 0, False)
    bar = pic[BAR_TOP : BAR_TOP + BAR_HEIGHT, BAR_MARGIN : BAR_MARGIN + BAR_WIDTH]
    assert (bar == bar[0]).all()  # every bar row identical
    assert bar[0, 0].tolist() == PALETTES[palette][0].tolist()
    assert bar[0, -1].tolist() == PALETTES[palette][255].tolist()
    lum = bar[0].astype(int).sum(axis=1)
    if palette != "ironbow":  # per-channel truncation gives ironbow 1-step dips between stops
        assert np.all(np.diff(lum) <= 0) if palette == "black-hot" else np.all(np.diff(lum) >= 0)
    assert (pic[SIDE:BAR_TOP] == PANEL).all() and (pic[SIDE:, :BAR_MARGIN] == PANEL).all()


def test_marker_colours_at_their_pixels_001_AC3() -> None:
    pic = grey_picture(readouts(hot=(10, 20), cold=(60, 70)))
    for (row, col), color in (((10, 20), HOT), ((60, 70), COLD)):
        y, x = row * SCALE + SCALE // 2, col * SCALE + SCALE // 2
        for py, px in ((y - HALF, x), (y + HALF - 1, x), (y, x - HALF), (y, x + HALF - 1)):
            assert pic[py, px].tolist() == list(color), (color, py, px)
        assert pic[y, x].tolist() == [GREY] * 3  # hollow square
    assert pic[C, C - HALF].tolist() == list(CENTRE) and pic[C - 1, C].tolist() == list(CENTRE)


def runs(line: Image) -> list[tuple[int, int]]:
    """(value, length) runs of a 1-D grey line."""
    out: list[tuple[int, int]] = []
    for v in line.tolist():
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1] + 1)
        else:
            out.append((v, 1))
    return out


def assert_even_stroke(line: Image, start: int, value: int) -> None:
    """`line` crosses one stroke centred on index `start`: GREY, dark, `value`, dark, GREY."""
    before, after = start - STROKE // 2 - OUTLINE, len(line) - start - STROKE // 2 - OUTLINE
    assert runs(line) == [(GREY, before), (DARK, OUTLINE), (value, STROKE), (DARK, OUTLINE), (GREY, after)]


def test_outline_even_001_AC3() -> None:
    pic = grey_picture(readouts(hot=(20, 20), cold=(70, 20)))[..., 0]
    # Crosshair: both arms across their width, then end to end along their length.
    assert_even_stroke(pic[C - 10 : C + 10, C - 12], 10, 255)
    assert_even_stroke(pic[C - 12, C - 10 : C + 10], 10, 255)
    arm = runs(pic[C - 1, C - HALF - 5 : C + HALF + 5])
    assert arm == [
        (GREY, 5 - OUTLINE),
        (DARK, OUTLINE),
        (255, 2 * HALF),
        (DARK, OUTLINE),
        (GREY, 5 - OUTLINE),
    ]
    # Where the two arms' outlines overlap (inner corners), darkened once, not twice.
    lo, hi = C - STROKE // 2 - 1, C + STROKE // 2
    assert pic[lo, lo] == DARK and pic[hi, hi] == DARK
    # Hot marker: each side, centred on the box edge.
    y, x = 20 * SCALE + SCALE // 2, 20 * SCALE + SCALE // 2
    assert_even_stroke(pic[y - HALF - 8 : y - HALF + 8, x], 8, HOT[0])
    assert_even_stroke(pic[y + HALF - 8 : y + HALF + 8, x], 8, HOT[0])
    assert_even_stroke(pic[y, x - HALF - 8 : x - HALF + 8], 8, HOT[0])


def test_marker_label_flips_at_the_right_edge_001_AC3() -> None:
    text = "30.0"
    gap = HALF + STROKE // 2 + OUTLINE + 2
    x_left = 50 * SCALE + SCALE // 2
    assert marker_label_x(x_left, text) == x_left + gap
    x_right = 95 * SCALE + SCALE // 2
    assert marker_label_x(x_right, text) == x_right - gap - badge_width(text)
    # Last column where the label still fits on the right, then the first that flips.
    fit = SIDE - gap - badge_width(text)
    assert marker_label_x(fit, text) == fit + gap and marker_label_x(fit + 1, text) < fit
    # In the picture: the badge backdrop sits left of a right-edge marker, right of a left one.
    pic = grey_picture(readouts(hot=(40, 95), cold=(40, 10)))[..., 0]
    y = 40 * SCALE + SCALE // 2
    row = pic[y - HALF + 2]
    assert (row[x_right - gap - badge_width(text) : x_right - gap] < GREY).any()
    xc = 10 * SCALE + SCALE // 2
    assert (row[xc + gap : xc + gap + badge_width(text)] < GREY).any()


def test_centre_badge_lower_right() -> None:
    f = font(FONT_SIZE)
    pic = grey_picture(readouts())[..., 0]
    x0 = y0 = C + SCALE + 2
    box = pic[y0 : y0 + f.line, x0 : x0 + badge_width("20.0")]
    assert box.max() == 255 and (box == GREY // 4).any()  # white digits on the dark backdrop
    assert pic[y0 - 1, x0 + 30] == GREY and pic[y0 + f.line, x0 + 30] == GREY


def test_scale_labels_001_AC3() -> None:
    labels = scale_labels(20.0, 25.0, 30.0)
    assert [t for t, _ in labels] == ["20.0", "25.0", "30.0°C"]
    assert labels[0][1] == BAR_MARGIN
    assert labels[2][1] + text_width("30.0°C", FONT_SIZE) == BAR_MARGIN + BAR_WIDTH
    assert labels[1][1] == SIDE // 2 - text_width("25.0", FONT_SIZE) // 2
    for lo, hi in ((-40.0, 400.0), (-999.9, 999.9), (0.0, 0.1)):
        spans = [(x, x + text_width(t, FONT_SIZE)) for t, x in scale_labels(lo, (lo + hi) / 2, hi)]
        assert spans[0][0] >= 0 and spans[-1][1] <= SIDE
        assert spans[0][1] < spans[1][0] and spans[1][1] < spans[2][0]


def test_scale_labels_drawn_in_the_band() -> None:
    band = grey_picture(readouts())[BAR_TOP + BAR_HEIGHT :, :, 0]
    assert band.max() == 255 and band.min() == PANEL
    cols = np.flatnonzero((band > 128).any(axis=0))
    assert (
        BAR_MARGIN <= cols[0] < BAR_MARGIN + 4 and BAR_MARGIN + BAR_WIDTH - 4 < cols[-1] < SIDE - BAR_MARGIN
    )
    assert ((band > 128).any(axis=1)).nonzero()[0][-1] < band.shape[0] - 1  # ink above the bottom edge


@pytest.mark.parametrize(
    ("c", "want"),
    [
        (23.04, "23.0"),
        (23.05, "23.1"),
        (-0.04, "0.0"),
        (-0.05, "-0.1"),
        (-12.34, "-12.3"),
        (0.0, "0.0"),
        # Java Math.round: NaN -> 0, out of range -> Long.MAX_VALUE tenths.
        (float("nan"), "0.0"),
        (-float("nan"), "0.0"),
        (float("inf"), "922337203685477580.7"),
        (-float("inf"), "-922337203685477580.7"),
        (1e300, "922337203685477580.7"),
    ],
)
def test_format_celsius(c: float, want: str) -> None:
    assert format_celsius(c) == want


def test_capture_markers_and_labels(capture: bytes) -> None:
    """Frame 0 of the bundled capture: markers framed, labels read from the readouts."""
    shown, r = Processor().process(frame(capture, 0), 0.0)
    pic = compose(shown, r, "ironbow", 0.0, 0, False)
    for spot, color in ((r.hot, HOT), (r.cold, COLD)):
        y, x = spot.row * SCALE + SCALE // 2, spot.col * SCALE + SCALE // 2
        top = max(y - HALF, STROKE // 2)
        left = max(x - HALF, STROKE // 2)
        assert pic[top, x].tolist() == list(color) or pic[y, left].tolist() == list(color)


# Text and PNG -----------------------------------------------------------------------------------


def test_glyph_metrics() -> None:
    f = font(FONT_SIZE)
    assert f.advance == 12 and f.ascent == 19 and f.line == 23
    rows = {ch: np.flatnonzero(f.glyphs[ch].max(axis=1) > 0.5) for ch in "0.C-°"}
    assert all(rows[ch][-1] + 1 == f.baseline for ch in "0.C")  # each glyph sits on the shared baseline
    assert rows["°"][0] == rows["0"][0] and rows["0"][0] < rows["-"][0] < rows["-"][-1] < rows["."][0]
    assert set(f.glyphs) == set(text.CHARS)


def test_draw_text_clipped_and_blended() -> None:
    img = np.full((40, 80, 3), 200, np.uint8)
    draw_text(img, 2, 25, "-1.5°C", FONT_SIZE, (255, 0, 0))
    assert (img == (255, 0, 0)).all(axis=-1).any() and (img[..., 1] < 200).any()
    edge = img.copy()
    draw_text(edge, -30, 5, "88", FONT_SIZE, (0, 0, 0))  # mostly outside: no crash, partly drawn
    draw_text(edge, 500, 25, "88", FONT_SIZE, (0, 0, 0))  # fully outside: untouched
    assert (edge[:, 40:] == img[:, 40:]).all()


def test_font_missing_is_readable(monkeypatch: pytest.MonkeyPatch, fresh_font: None) -> None:
    monkeypatch.setattr(text, "FONT_FILE", Path("/nonexistent/DejaVuSansMono.ttf"))
    with pytest.raises(FontError, match="fonts-dejavu-core"):
        font(FONT_SIZE)


def test_ffmpeg_missing_is_readable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh_font: None
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FontError, match="ffmpeg not found"):
        font(FONT_SIZE)


def test_ffmpeg_timeout_is_readable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh_font: None
) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nexec sleep 10\n")  # exec: the timeout kill must reach the sleeper
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setattr(text, "FFMPEG_TIMEOUT", 0.2)
    with pytest.raises(FontError, match="ffmpeg drawtext timed out after 0.2 s"):
        font(FONT_SIZE)


@pytest.mark.parametrize(
    ("script", "message"),
    [
        pytest.param(
            "echo 'No such filter: drawtext' >&2; exit 1", "drawtext failed: No such filter", id="exit-1"
        ),
        pytest.param("printf abc", "returned 3 bytes", id="short-output"),
        pytest.param(
            f"head -c {(2 * FONT_SIZE) ** 2 * len(text.CHARS)} /dev/zero", "drew no glyphs", id="blank-output"
        ),
    ],
)
def test_ffmpeg_failure_is_readable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh_font: None, script: str, message: str
) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text(f"#!/bin/sh\n{script}\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    with pytest.raises(FontError, match=message):
        font(FONT_SIZE)


def test_png_roundtrip(capture: bytes) -> None:
    shown, r = Processor().process(frame(capture, 0), 0.0)
    pic = compose(shown, r, "ironbow", 0.0, 0, False)
    assert np.array_equal(decode_png(encode_png(pic)), pic)


def test_readout_panel_001_AC4() -> None:
    """Min / Max / Center are the cold / hot / centre badge values, each in its colour; Center the largest."""
    r = Readouts(Spot(3, 4, 41.25), Spot(5, 6, -3.25), 24.04, 0.0, 50.0, False)
    assert readout_texts(r) == ("-3.3", "41.3", "24.0")
    blank, panel = readout_panel(None), readout_panel(r)
    assert blank.shape == panel.shape and blank.shape[1] == WIDTH and not blank.any()
    half = WIDTH // 2
    for color, cols in ((COLD, slice(0, half)), (HOT, slice(half, WIDTH)), (CENTRE, slice(0, WIDTH))):
        assert (panel[:, cols] == color).all(axis=-1).any(), color
    assert not (panel[:, half:] == COLD).all(axis=-1).any()
    assert not (panel[:, :half] == HOT).all(axis=-1).any()
    assert CENTRE_SIZE > PAIR_SIZE > LABEL_SIZE
    # Swapped spots swap the numbers: Min follows the cold marker, not the smaller value.
    swapped = Readouts(Spot(0, 0, 1.0), Spot(0, 0, 2.0), 0.0, 0.0, 1.0, False)
    assert readout_texts(swapped)[:2] == ("2.0", "1.0")
