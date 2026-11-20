# tc001-mini-linux

A desktop viewer for the TOPDON TC001 Mini thermal camera on Linux. No vendor SDK, no network access. It mirrors the Android app [tc001-mini-android](https://github.com/babs/tc001-mini-android): same picture, same readouts, same files.

**Status: planned.** The behaviour to build is in [`specs/001-linux-viewer.md`](specs/001-linux-viewer.md); nothing is implemented yet.

## What it will do

- Live thermal image at the camera's 25 fps through the stock Linux UVC driver (`uvcvideo`, read with `v4l2-ctl`).
- Ironbow / white-hot / black-hot palettes, auto or locked colour range, rotation and mirror, display smoothing.
- Hot / cold spot markers and a centre crosshair, a horizontal colour scale, and large Min / Max / Center readouts.
- Photo (PNG of the picture with its readouts as JSON, plus the raw frame) and video (MP4).
- Buttons and keyboard shortcuts; settings kept between runs.

## How the camera is read

The TC001 Mini (USB `2bdf:0102`) enumerates as a UVC camera with one YUYV format whose odd frame sizes carry raw data. The `4x2321` size is "mode 2": a 136-byte header (magic `0x70827773` at 0, marker `0xAABBCCDD` at 132) followed by 96x96 little-endian `uint16`, with `celsius = raw / 64 - 50`.

```sh
v4l2-ctl -d /dev/videoN --set-fmt-video=width=4,height=2321,pixelformat=YUYV --stream-mmap --stream-to=-
```

Reading the device needs no root when your user can open `/dev/videoN` (usually the `video` group).

## Requirements

- Linux with `uvcvideo` and `v4l2-ctl` (package `v4l-utils`).
- Python 3.14 with Tk (`tkinter`), managed with [uv](https://docs.astral.sh/uv/).
- `ffmpeg` for video recording.

## License

[MIT](LICENSE)
