"""Tk window: live picture, Min / Max / Center readouts, controls and shortcuts, camera loss and return."""

import dataclasses
import math
import queue
import signal
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk
from typing import NamedTuple

import numpy as np

from tc001_linux import Image, Plane
from tc001_linux import settings as cfg
from tc001_linux.capture import Recorder, capture_name, media_dir, photo_json, save_photo
from tc001_linux.picture import CENTRE, COLD, FONT_SIZE, HEIGHT, HOT, WIDTH, compose, readout_texts
from tc001_linux.png import encode_png
from tc001_linux.process import Processor, Readouts
from tc001_linux.source import Mailbox, OnFrame, Source, Step, Watch
from tc001_linux.text import font

TITLE = "TC001 Mini"
TICK_MS = 10
READOUT_PERIOD = 0.2  # ~5 Hz, like the Android app: steadier to read than every frame
DEBUG_PERIOD = 1.0
STATUS_SECONDS = 4.0
PHOTO_WAIT = 1.0  # a photo asked this long ago with no frame yet gets a status line
BG, BUTTON_BG, FG, DIM, RED = "#000000", "#181818", "#ffffff", "#8a8a8a", "#ff3b30"
HOVER, PRESSED, TROUGH, THUMB = "#242424", "#303030", "#4a4a4a", "#d0ccff"
UI_FONT, MONO_FONT = "DejaVu Sans", "DejaVu Sans Mono"
UI_PX = 15
READOUT_PX = (20, 48, 64)  # px of the titles, Min / Max and Center at picture scale 1
RESIZE_MS = 80  # a drag sends a burst of <Configure>: lay out once it settles
MIN_SCALE = 0.4  # smallest picture scale the minimum window size allows
MAX_PICTURE_HEIGHT = 1600  # beyond, letterboxed: resampling cost grows with the area
SCREEN_MARGIN = 80  # room left for panels and the title bar when the natural size exceeds the screen
PALETTE_NAMES = {"ironbow": "Ironbow", "white-hot": "White hot", "black-hot": "Black hot"}
WAITING, DISCONNECTED = "Waiting for the camera…", "Camera disconnected"

type Connect = Callable[[OnFrame], Source | None]


def ppm(img: Image) -> bytes:
    h, w, _ = img.shape
    return f"P6 {w} {h} 255 ".encode() + img.tobytes()


class Shot(NamedTuple):
    """One processed frame: raw bytes, its numbers and its WIDTH x HEIGHT picture."""

    raw: bytes
    readouts: Readouts
    picture: Image


def pill(w: int, h: int, color: str) -> Image:
    """A w x h bar with round ends (a disc when w == h), anti-aliased onto the black background."""
    r = h / 2
    y, x = np.mgrid[:h, :w] + 0.5
    d = np.hypot(np.clip(x, r, w - r) - x, y - r)
    cover = np.clip(r - d + 0.5, 0, 1)[..., None]
    out: Image = (cover * np.frombuffer(bytes.fromhex(color[1:]), np.uint8) + 0.5).astype(np.uint8)
    return out


def theme(root: tk.Tk) -> list[tk.PhotoImage]:
    """Dark ttk theme on "clam", DejaVu Sans everywhere; returns the slider images, to be kept alive."""
    tkfont.nametofont("TkDefaultFont", root).configure(family=UI_FONT, size=-UI_PX)
    style = ttk.Style(root)
    style.theme_use("clam")
    flat = {"lightcolor": BUTTON_BG, "darkcolor": BUTTON_BG}
    style.configure(".", background=BG, foreground=FG, font="TkDefaultFont", focuscolor=BG, **flat)
    style.configure("TButton", background=BUTTON_BG, padding=(8, 7), relief="flat")
    # clam draws the bevel with light / dark colours: they follow the background to stay flat.
    states = [("pressed", PRESSED), ("active", HOVER)]
    style.map(
        "TButton", background=states, lightcolor=states, darkcolor=states, foreground=[("disabled", DIM)]
    )
    style.configure("Recording.TButton", foreground=RED)
    style.map("Recording.TButton", foreground=[("active", RED)])
    images = [
        tk.PhotoImage(master=root, data=ppm(pill(*size, c)))
        for size, c in [((24, 4), TROUGH), ((18, 18), THUMB), ((18, 18), FG), ((18, 18), "#a8a2ff")]
    ]
    trough, thumb, hover, pressed = images
    style.element_create("Thin.trough", "image", trough, border=(2, 0), sticky="ew")
    style.element_create("Round.slider", "image", thumb, ("pressed", pressed), ("active", hover))
    style.layout(
        "Round.Horizontal.TScale",
        [("Thin.trough", {"sticky": "ew", "children": [("Round.slider", {"side": "left", "sticky": ""})]})],
    )
    return images


class App:
    def __init__(
        self,
        root: tk.Tk,
        connect: Connect,
        settings: cfg.Settings,
        settings_path: Path,
        fixed: tuple[float, float] | None = None,
        debug: bool = False,
    ) -> None:
        self.root, self.connect, self.settings, self.settings_path = root, connect, settings, settings_path
        self.proc = Processor(fixed)
        self.mailbox = Mailbox[Shot]()
        self.watch = Watch()
        self.source: Source | None = None
        self.shown = 0  # pictures put on screen
        self.readout_shot: Shot | None = None  # frame whose numbers the readouts show
        self.readout_texts: tuple[str, str, str] | None = None
        self._next_readout = 0.0
        self._quit = False
        self._unsaved = False  # slider moved, saved on release
        self._errors: set[str] = set()  # source errors already printed
        self._ui_ms = 0.0
        self._fps_at, self._fps_frames = time.monotonic(), 0
        self._photo = threading.Event()  # taken by the reader on the next frame
        self.pictures, self.videos = media_dir("PICTURES"), media_dir("VIDEOS")  # xdg-user-dir runs once
        self._photo_at = math.inf  # when the pending photo was asked, until its wait is reported
        self.recorder: Recorder | None = None
        self._statuses = queue.SimpleQueue[str]()  # from the photo and recorder threads
        self._status_until = 0.0

        self._shown_picture: Image | None = None  # last picture on screen, redrawn at a new size
        self._size = (WIDTH, HEIGHT)
        self._index: np.ndarray[tuple[int], np.dtype[np.int32]] | None = None  # pixel map, None at scale 1
        self._layout_id: str | None = None
        self._laid_out: tuple[int, int] | None = None  # window size of the last layout

        root.title(TITLE)
        root.configure(bg=BG)
        self._images = theme(root)
        # Packed bottom-up, the view last: a shrinking window squeezes the picture, never the controls.
        self.status = ttk.Label(root, anchor="w", padding=(8, 2))
        self.status.pack(side="bottom", fill="x")
        slider = ttk.Frame(root, padding=(8, 4))
        slider.pack(side="bottom", fill="x")
        rows = [ttk.Frame(root, padding=(2, 0)) for _ in range(2)]
        for row in reversed(rows):
            row.pack(side="bottom", fill="x")
        self.readouts = ttk.Frame(root, padding=(0, 4))
        self.readouts.pack(side="bottom")  # as wide as the picture: Min / Max sit under its quarters
        self.view = tk.Frame(root, bg=BG, width=WIDTH, height=HEIGHT)
        self.view.pack_propagate(False)  # its size comes from the window, not from the picture
        self.view.pack(side="top", fill="both", expand=True)
        self.photo = tk.PhotoImage(master=root, width=WIDTH, height=HEIGHT)
        self.picture = tk.Label(self.view, image=self.photo, bg=BG, bd=0, highlightthickness=0)
        self.picture.pack(expand=True)  # centred, rounded down: place could push it 1 px past the view
        self.message = tk.Label(root, text=WAITING, font=(UI_FONT, -20), fg=FG, bg=BG, padx=16, pady=8)
        self.message.place(in_=self.picture, relx=0.5, rely=0.5, anchor="center")
        self.debug = tk.Label(root, font=(MONO_FONT, -12), fg=FG, bg=BG) if debug else None
        if self.debug:
            self.debug.place(in_=self.picture, x=0, y=0)

        # Tk labels in DejaVu Sans Mono: Xft renders them sharp at any size, so they scale with the picture.
        self.readout_fonts = [tkfont.Font(root, family=MONO_FONT, size=-px) for px in READOUT_PX]
        title, pair, centre = self.readout_fonts
        self.readouts.columnconfigure((0, 1), minsize=WIDTH // 2)
        cells = [
            (0, 0, 1, "Min", title, COLD),
            (0, 1, 1, "Max", title, HOT),
            (2, 0, 2, "Center", title, CENTRE),
            (1, 0, 1, "", pair, COLD),
            (1, 1, 1, "", pair, HOT),
            (3, 0, 2, "", centre, CENTRE),
        ]
        labels = []
        for r, col, span, text, f, color in cells:
            labels.append(ttk.Label(self.readouts, text=text, font=f, foreground="#" + bytes(color).hex()))
            labels[-1].grid(row=r, column=col, columnspan=span)
        self.min_label, self.max_label, self.centre_label = labels[3:]

        def button(row: ttk.Frame, text: str, cmd: Callable[[], None]) -> ttk.Button:
            b = ttk.Button(row, text=text, command=cmd, takefocus=False)
            b.pack(side="left", expand=True, fill="x", padx=2, pady=2)
            return b

        self.photo_button = button(rows[0], "Photo", lambda: self.act("photo"))
        self.record_button = button(rows[0], "", lambda: self.act("record"))
        self.palette_button = button(rows[1], "", lambda: self.act("palette"))
        self.lock_button = button(rows[1], "", lambda: self.act("lock"))
        self.rotate_button = button(rows[1], "", lambda: self.act("rotate"))
        self.mirror_button = button(rows[1], "", lambda: self.act("mirror"))
        ttk.Label(slider, text="Smoothing").pack(side="left")
        self.smoothing = tk.DoubleVar(root, settings.smoothing)
        self.smoothing_value = ttk.Label(slider, width=4, anchor="e")
        self.smoothing_value.pack(side="right")
        self.scale = ttk.Scale(
            slider,
            variable=self.smoothing,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            command=self._slid,
            style="Round.Horizontal.TScale",
            takefocus=False,
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=12)
        self.scale.bind("<ButtonRelease-1>", lambda e: self._flush())  # ttk runs -command on each motion
        self.controls = (self.status, slider, *rows)
        self._label_controls()

        root.update_idletasks()  # requested sizes, for the minimum window size
        controls = sum(c.winfo_reqheight() for c in self.controls)
        content = HEIGHT + self.readouts.winfo_reqheight()
        root.minsize(max(r.winfo_reqwidth() for r in rows), controls + math.ceil(MIN_SCALE * content))
        # The natural height is over 1080 px: on such a screen the controls would be off it.
        sw, sh = root.winfo_screenwidth() - SCREEN_MARGIN, root.winfo_screenheight() - SCREEN_MARGIN
        rw, rh = root.winfo_reqwidth(), root.winfo_reqheight()
        if rw > sw or rh > sh:
            root.geometry(f"{min(rw, sw)}x{min(rh, sh)}")
        root.bind("<Configure>", self._configured)
        root.bind("<KeyPress>", lambda e: self.act(cfg.key_action(e.keysym) or ""))
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self._tick_id = root.after(0, self._tick)

    # Reader thread: every frame goes through the tracker, only the latest picture reaches the screen.
    def on_frame(self, raw: bytes, t: Plane) -> None:
        s = self.settings
        shown, r = self.proc.process(t, s.smoothing)
        img = compose(shown, r, s.palette, s.smoothing, s.rotation, s.mirror)
        self.mailbox.put(Shot(raw, r, img))
        if self._photo.is_set():
            self._photo.clear()
            args = (img, raw, r, s, time.time_ns() // 1_000_000)
            threading.Thread(target=self._save_photo, args=args, name="photo").start()
        if (rec := self.recorder) is not None:
            rec.feed(img.tobytes())

    # Photo thread: PNG encoding and disk writes stay off the Tk thread.
    def _save_photo(self, img: Image, raw: bytes, r: Readouts, s: cfg.Settings, epoch_ms: int) -> None:
        name = capture_name(epoch_ms)
        try:
            png = encode_png(img, ("tc001", photo_json(r, s, epoch_ms)))
            save_photo(self.pictures, name, png, raw)
            message = f"Photo saved: {name}"
        except (OSError, ValueError) as e:
            message = f"Photo failed: {e}"
        self.report(message)

    def report(self, text: str) -> None:
        """Any thread: a status line for STATUS_SECONDS, also printed."""
        self._statuses.put(text)
        try:
            print(f"tc001-mini-linux: {text}", file=sys.stderr)
        except OSError:  # stderr gone (closed terminal): the status line still shows it
            pass

    def _record(self) -> None:
        if self.recorder is not None:
            self.recorder.stop()
            self.recorder = None
        elif self.source is not None:  # without a source no frame would ever come
            try:
                name = capture_name(time.time_ns() // 1_000_000)
                self.recorder = Recorder(self.videos, name, self.report)
            except OSError as e:
                self.report(f"Video failed: {e}")
        self._label_controls()

    def act(self, action: str) -> None:
        if action == "quit":
            self.quit()
        elif action == "photo":
            if self.source is None:
                self.report("Photo failed: no frame")
            else:
                self._photo_at = time.monotonic()
                self._photo.set()
        elif action == "record":
            self._record()
        elif action == "lock":
            if self.proc.fixed is None:
                self.proc.locked = not self.proc.locked
                self._label_controls()
        else:
            self._set(cfg.apply(self.settings, action))

    def _slid(self, value: str) -> None:
        """Applied live in 0.01 steps; a drag calls this on every motion, so the file is saved on release."""
        if (v := round(float(value), 2)) != self.settings.smoothing:
            self._set(dataclasses.replace(self.settings, smoothing=v), save=False)

    def _set(self, s: cfg.Settings, save: bool = True) -> None:
        if s == self.settings:
            return
        self.settings = s
        self._label_controls()
        self._unsaved = True
        if save:
            self._flush()

    def _flush(self) -> None:
        if not self._unsaved:
            return
        self._unsaved = False
        try:
            cfg.save(self.settings_path, self.settings)
        except OSError as e:
            print(f"tc001-mini-linux: settings not saved: {e}", file=sys.stderr)

    def _label_controls(self) -> None:
        s = self.settings
        self.palette_button.configure(text=PALETTE_NAMES[s.palette])
        if self.proc.fixed is not None:  # --range: the lock has nothing to hold
            self.lock_button.configure(text="Fixed range", state="disabled")
        else:
            self.lock_button.configure(text="Range locked" if self.proc.locked else "Auto range")
        self.rotate_button.configure(text=f"Rotate {s.rotation}°")
        self.mirror_button.configure(text="Mirrored" if s.mirror else "Mirror off")
        on = self.recorder is not None
        self.record_button.configure(
            text="Stop" if on else "Record", style="Recording.TButton" if on else "TButton"
        )
        if self.smoothing.get() != s.smoothing:
            self.smoothing.set(s.smoothing)
        self.smoothing_value.configure(text=f"{s.smoothing:.2f}")

    def drop_source(self) -> None:
        """Stops the source, and the recording: no frame would come."""
        if self.recorder is not None:
            self.recorder.stop()
            self.recorder = None
        if self._photo.is_set():
            self._photo.clear()
            self.report("Photo failed: no frame")
        if self.source is not None:
            self.source.stop()
            self.source = None
        self.mailbox.take()  # a frame put before the reader stopped

    def _tick(self) -> None:
        if self._quit:
            self.quit()
            return
        now = time.monotonic()
        shot = self.mailbox.take()
        if shot is not None:
            t0 = time.perf_counter()
            self._draw(shot.picture)
            self._ui_ms += 0.1 * ((time.perf_counter() - t0) * 1000 - self._ui_ms)
            self.shown += 1
            self.message.place_forget()
            if now >= self._next_readout:
                self._show_readouts(shot)
                self._next_readout = now + READOUT_PERIOD
        match self.watch.step(now, shot is not None, self.source is not None and self.source.alive):
            case Step.LOST:
                lost = self.source
                self.drop_source()
                error = lost.error if lost else None
                if error and error not in self._errors:
                    self._errors.add(error)
                    print(f"tc001-mini-linux: {error}", file=sys.stderr)
                self.photo.blank()  # stale temperatures must not pass for live ones
                self._shown_picture = None
                self._label_controls()
                self._show_readouts(None)
                self._fps_frames = 0
                self.message.configure(text=f"{DISCONNECTED}\n{error}" if error else DISCONNECTED)
                self.message.place(in_=self.picture, relx=0.5, rely=0.5, anchor="center")
            case Step.POLL:
                try:
                    self.source = self.connect(self.on_frame)
                except OSError as e:  # an exception here would end the tick loop and freeze the window
                    print(f"tc001-mini-linux: cannot open the source: {e}", file=sys.stderr)
                if self.source is not None:
                    self.watch.started(now)
                    self._fps_at, self._fps_frames = now, 0
            case Step.OK:
                pass
        if self._photo.is_set() and now - self._photo_at >= PHOTO_WAIT:
            self._photo_at = math.inf  # once per photo
            self.report("Photo: waiting for a frame")
        if self.recorder is not None and self.recorder.stopped:  # ffmpeg ended by itself
            self.recorder = None
            self._label_controls()
        while not self._statuses.empty():
            self.status.configure(text=self._statuses.get())
            self._status_until = now + STATUS_SECONDS
        if self._status_until and now >= self._status_until:
            self.status.configure(text="")
            self._status_until = 0.0
        if self.debug and now - self._fps_at >= DEBUG_PERIOD:
            frames = self.source.frames if self.source else 0
            rejected = self.source.rejected if self.source else 0
            fps = (frames - self._fps_frames) / (now - self._fps_at)
            w, h = self._size
            self.debug.configure(
                text=f"{fps:.1f} fps, rejected: {rejected}, ui {self._ui_ms:.1f} ms at {w}x{h}"
            )
            self._fps_at, self._fps_frames = now, frames
        self._tick_id = self.root.after(TICK_MS, self._tick)

    def _show_readouts(self, shot: Shot | None) -> None:
        self.readout_shot = shot
        self.readout_texts = readout_texts(shot.readouts) if shot else None
        labels = (self.min_label, self.max_label, self.centre_label)
        for label, text in zip(labels, self.readout_texts or ("", "", ""), strict=True):
            label.configure(text=text)

    def _draw(self, picture: Image) -> None:
        """Puts `picture` on screen at the display size: nearest neighbour through the precomputed map."""
        self._shown_picture = picture
        if self._index is not None:
            w, h = self._size
            picture = np.take(picture.reshape(-1, 3), self._index, axis=0).reshape(h, w, 3)
        self.photo.put(ppm(picture))

    def _configured(self, event: tk.Event) -> None:
        if event.widget is self.root and (event.width, event.height) != self._laid_out:  # not a move
            if self._layout_id is not None:
                self.root.after_cancel(self._layout_id)
            self._layout_id = self.root.after(RESIZE_MS, self._layout)

    def _layout(self) -> None:
        """Largest picture scale that fits the window above the controls; the readouts follow it."""
        self._layout_id = None
        w, h = self.root.winfo_width(), self.root.winfo_height()
        self._laid_out = (w, h)
        fit = min(w, WIDTH * MAX_PICTURE_HEIGHT / HEIGHT)  # width the picture may take
        room = h - sum(c.winfo_reqheight() for c in self.controls)
        scale = self._size[0] / WIDTH
        # The readouts grow with the scale and take height from the picture: start from a proportional
        # estimate, then measure; each pass shrinks the error about fourfold (readouts ~1/4 of the picture).
        scale = min(fit / WIDTH, room / (HEIGHT + self.readouts.winfo_reqheight() / scale))
        for _ in range(2):
            for f, px in zip(self.readout_fonts, READOUT_PX, strict=True):
                f.configure(size=-max(1, round(px * scale)))
            self.root.update_idletasks()
            scale = min(fit / WIDTH, (room - self.readouts.winfo_reqheight()) / HEIGHT)
        self._resize(max(1, math.floor(WIDTH * scale)))

    def _resize(self, w: int) -> None:
        h = round(w * HEIGHT / WIDTH)
        if (w, h) == self._size:
            return
        self._size = (w, h)
        rows, cols = np.arange(h, dtype=np.int32) * HEIGHT // h, np.arange(w, dtype=np.int32) * WIDTH // w
        self._index = None if (w, h) == (WIDTH, HEIGHT) else (rows[:, None] * WIDTH + cols).ravel()
        self.photo.configure(width=w, height=h)
        self.readouts.columnconfigure((0, 1), minsize=w // 2)
        if self._shown_picture is not None:
            self._draw(self._shown_picture)
        else:
            self.photo.blank()

    def request_quit(self) -> None:
        """Safe from a signal handler: the next tick quits."""
        self._quit = True

    def quit(self) -> None:
        self._flush()
        self.drop_source()
        # A callback firing after destroy() prints "invalid command name".
        self.root.after_cancel(self._tick_id)
        if self._layout_id is not None:
            self.root.after_cancel(self._layout_id)
        self.root.destroy()


def run(
    connect: Connect,
    settings: cfg.Settings,
    settings_path: Path,
    fixed: tuple[float, float] | None = None,
    debug: bool = False,
) -> None:
    """Open the window and run until it is closed; FontError propagates, a missing display exits."""
    font(FONT_SIZE)  # rasterised once here, before the reader thread composes pictures
    try:
        root = tk.Tk(className="tc001-mini-linux")
    except tk.TclError as e:
        sys.exit(f"tc001-mini-linux: cannot open the window: {e}")
    app = App(root, connect, settings, settings_path, fixed, debug)

    def quit_soon(*_: object) -> None:
        app.request_quit()

    # SIGHUP: the terminal closed; quitting cleanly finalises a recording.
    previous = {sig: signal.signal(sig, quit_soon) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        root.mainloop()
    finally:
        app.drop_source()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
