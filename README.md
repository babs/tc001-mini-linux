# tc001-mini-linux

A desktop viewer for the TOPDON TC001 Mini thermal camera on Linux, with no vendor SDK and no network access. It mirrors the Android app [tc001-mini-android](https://github.com/babs/tc001-mini-android): the same picture (thermal image, colour scale, hot / cold markers, centre crosshair), large Min / Max / Center readouts, palette, range lock, rotation, mirror and smoothing controls, and photos and videos in the same formats, so files from both read the same.

## Status

Proof of concept, kept local. Live view, controls, saved settings, photo and video work; window polish (dark theme, resizable window) is next. The plan and acceptance criteria are in [`specs/001-linux-viewer.md`](specs/001-linux-viewer.md).

## Usage

Prerequisites:
- System packages: `sudo apt install python3-tk v4l-utils ffmpeg fonts-dejavu-core`
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (the project runs on the system Python 3.14 and its Tk)
- Camera access without root: `sudo usermod -aG video "$USER"`, then log out and back in
- Check the camera is seen: `v4l2-ctl --list-devices | grep -A3 -i camera`

Install and run:
- Install from a clone: `uv tool install --python-preference only-system .`
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

## How the camera is read

The TC001 Mini (USB `2bdf:0102`) enumerates as a UVC camera whose YUYV frame size `4x2321` carries raw data ("mode 2"): a 136-byte header (magic `0x70827773` at 0, marker `0xAABBCCDD` at 132), then 96x96 little-endian `uint16`, with `celsius = raw / 64 - 50`. The viewer reads it through the stock `uvcvideo` driver with `v4l2-ctl`, so the `video` group is enough:

```sh
v4l2-ctl -d /dev/videoN --set-fmt-video=width=4,height=2321,pixelformat=YUYV --stream-mmap --stream-to=-
```

A photo's `.bin` is one such 18,568-byte frame, untouched. The PNG holds a `tEXt` chunk `tc001` with the readouts and view settings as JSON, as the Android app writes it.

## License

[MIT](LICENSE)
