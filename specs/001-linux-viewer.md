# 001 — TC001 Mini Linux viewer

**Status**: in progress
**Requested by**: babs
**Date**: 2026-11-20

## Problem

On Linux the only way to look through the TC001 Mini is `scripts/live_view.py`, a proof of concept outside any repository: an `ffplay` window with no buttons, no photo or video, settings only as command-line flags. The Android app [tc001-mini-android](https://github.com/babs/tc001-mini-android) has the interface the owner wants; the desktop has nothing comparable.

## Solution

A small desktop app: plug the camera in, run `tc001-mini-linux`, and a window shows the same picture as the Android app (thermal image, horizontal colour scale, hot / cold markers, centre crosshair) with large Min / Max / Center readouts and buttons for the palette, range lock, rotation, mirror, smoothing, photo and video. Photos and videos land in the user's Pictures and Videos folders, in the same formats as the Android app.

## Scope

- Python 3.14 package managed with uv, one runtime dependency (numpy), UI in Tk (standard library). Console entry point `tc001-mini-linux`.
- Source: the camera through `uvcvideo` + `v4l2-ctl` (mode 2, `4x2321`), auto-detected by USB id `2bdf:0102`, or `--device`; `--file` replays a raw mode-2 capture (looped).
- Picture identical in layout to the Android app: 768x768 thermal image (scale 8), 64 px band with the horizontal scale and low / middle / high labels with °C; cold marker cyan, hot marker pink-red, centre badge white; crosshair and boxes on a thin black outline.
- Window: the picture, then Min | Max on one row and Center below in large type (colours as above, refreshed about 5 times per second), then the controls.
- Controls, each with a keyboard shortcut: palette (ironbow / white-hot / black-hot), lock range, rotate 90°, mirror, smoothing slider (0–1), Photo, Record / Stop.
- Settings kept across runs in `$XDG_CONFIG_HOME/tc001-mini-linux/settings.json`: palette, rotation, mirror, smoothing.
- Photo: `TC001_<timestamp>.png` (the picture, with a `tEXt` chunk `tc001` holding the same JSON as the Android app) in `$XDG_PICTURES_DIR/TC001` (default `~/Pictures/TC001`), and the raw frame `TC001_<timestamp>.bin` next to it.
- Video: `TC001_<timestamp>.mp4` (H.264, no audio, ~25 fps) of the picture through an `ffmpeg` subprocess, in `$XDG_VIDEOS_DIR/TC001`; finalised on Stop, window close and camera loss.
- Camera unplugged: "Camera disconnected" shown, recording finalised, the view resumes when the camera comes back (polls for the device, no udev rule needed).
- The maths of spec 001 of the Android app, ported: frame parsing, 3x3 spot tracker with 0.3 °C hysteresis and label EMA, display-only smoothing, auto / locked range.

## Usage (README content)

The README carries this section as is, one line per item.

Prerequisites:
- System packages: `sudo apt install v4l-utils ffmpeg fonts-dejavu-core`
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (provides Python 3.14 with Tk)
- Camera access without root: `sudo usermod -aG video "$USER"`, then log out and back in
- Check the camera is seen: `v4l2-ctl --list-devices | grep -A3 -i camera`

Install and run:
- Install from a clone: `uv tool install .`
- Run on the camera: `tc001-mini-linux`
- Pick the device: `tc001-mini-linux --device /dev/video2`
- Replay a capture: `tc001-mini-linux --file capture.bin`
- Render one frame without a window: `tc001-mini-linux --file capture.bin --snapshot out.png`
- Run from the clone without installing: `uv run tc001-mini-linux`

Keyboard:
- `P` photo, `R` record / stop, `C` next palette, `L` lock range, `T` rotate 90°, `M` mirror, `+` / `-` smoothing, `Q` quit

Files:
- Photos: `~/Pictures/TC001/TC001_<timestamp>.png` and `.bin` (or `$XDG_PICTURES_DIR/TC001`)
- Videos: `~/Videos/TC001/TC001_<timestamp>.mp4` (or `$XDG_VIDEOS_DIR/TC001`)
- Settings: `~/.config/tc001-mini-linux/settings.json` (or `$XDG_CONFIG_HOME/tc001-mini-linux`)
- Read a photo's readouts: `exiftool -Tc001 TC001_<timestamp>.png`

## Out of scope

- Packaging beyond `uv tool install` (no .deb, Flatpak, AppImage).
- Publishing: the repository stays local while this is a proof of concept.
- Windows and macOS.
- Emissivity / distance / ambient corrections (XU mailbox protocol not decoded).
- Other TOPDON models, other stream modes.
- Sharing code with the Android app (different language; the maths is small and ported).

## Acceptance criteria

- [ ] **AC1** With the camera plugged in and the user in the `video` group, `tc001-mini-linux` shows the live image within 3 s at about 25 fps, with no root and no udev rule.
- [ ] **AC2** Temperatures equal `raw/64 - 50`: unit test on the bundled capture (frame 0 min / max / mean 22.70 / 23.69 / 23.04 °C ± 0.01).
- [ ] **AC3** The picture is 768 x 832 with the layout and colours of the Android app's spec 002 AC1; for the bundled capture it shows the same layout, marker positions and values as the Android rendering of the same frame and settings (not pixel identical: fonts and edge clipping may differ).
- [ ] **AC4** Min, Max and Center in the window equal the cold marker, hot marker and centre badge values of the same frame; Center is larger than Min / Max.
- [ ] **AC5** Every control and its shortcut changes the view on the next frame; palette, rotation, mirror and smoothing survive a restart; lock range freezes the scale labels and colours.
- [ ] **AC6** Photo (button or shortcut) writes the PNG (768 x 832, `tEXt` `tc001` JSON with the Android field set) and the 18,568-byte `.bin` of the same frame; `exiftool -Tc001` reads the JSON.
- [ ] **AC7** Record / Stop writes a playable H.264 MP4 at 768 x 832, ~25 fps, duration within 1 s of the recording; closing the window or unplugging the camera finalises it.
- [ ] **AC8** Unplugging the camera shows "Camera disconnected" without crashing; plugging it back resumes the view within 5 s.
- [ ] **AC9** `--file <capture>` replays a capture so the whole UI works without a camera; the unit tests and a headless render check run in CI without a display.
- [ ] **AC10** The README holds the Usage section above, one line per item, and every command in it works as written on a fresh clone (checked by running each one).

## Phases

### Phase 1 — Core and picture
- Work: uv project, numpy-only core (parse, tracker, smoothing, palettes, orientation, readouts, picture composer incl. text and outline), bundled capture, unit tests, `--snapshot OUT.png` headless render.
- **Covers**: AC2, AC3, AC9 (tests part)
- **Data model impact**: none
- **DoD**: `uv run pytest` green; `uv run tc001-mini-linux --file <capture> --snapshot out.png` writes the picture; side-by-side with an Android photo of the same capture frame looks the same.

### Phase 2 — Window, controls, settings, live camera
- Work: Tk window with picture, big readouts and controls, shortcuts, settings file, camera source (v4l2-ctl), replay source, disconnect / reconnect.
- **Covers**: AC1, AC4, AC5, AC8, AC9
- **Data model impact**: settings file only
- **DoD**: live on the real camera at ~25 fps (frame counter in a debug flag), unplug / replug resumes, settings survive a restart, screenshot of the window.

### Phase 3 — Photo and video
- Work: PNG writer with `tEXt` (zlib, no new dependency), `.bin`, ffmpeg recorder, XDG folders; README Usage section.
- **Covers**: AC6, AC7, AC10
- **Data model impact**: none
- **DoD**: photo and video checked with `exiftool`, `ffprobe`; unplug during recording leaves a playable file.

## Data model impact (summary)

None, besides the settings JSON file.

## Open questions

- (none)

## Decisions

- `scripts/live_view.py` (the ffplay proof of concept) stays as the zero-install fallback once this ships (user choice).
- Tk over Qt / GTK: in the standard library (Tk 8.6 verified in uv's Python 3.14 and the system Python), real buttons and shortcuts, no 200 MB dependency.
- Camera through `uvcvideo` + `v4l2-ctl` rather than raw USB like Android: the Linux kernel driver already frames the stream and needs no extra permission beyond the `video` group.
- Same picture, readouts, colours and file formats as the Android app so photos from both are interchangeable.
