"""Tk window: live picture, Min / Max / Center readouts, controls and shortcuts, camera loss and return."""

import dataclasses
import signal
import sys
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from tc001_linux import Image, Plane
from tc001_linux import settings as cfg
from tc001_linux.picture import (
    CENTRE_SIZE,
    HEIGHT,
    LABEL_SIZE,
    PAIR_SIZE,
    WIDTH,
    compose,
    readout_panel,
    readout_texts,
)
from tc001_linux.process import Processor, Readouts
from tc001_linux.source import Mailbox, OnFrame, Source, Step, Watch
from tc001_linux.text import font

TITLE = "TC001 Mini"
TICK_MS = 10
READOUT_PERIOD = 0.2  # ~5 Hz, like the Android app: steadier to read than every frame
DEBUG_PERIOD = 1.0
BG, BUTTON_BG, FG, DIM = "#000000", "#1e1e1e", "#ffffff", "#777777"
UI_FONT = ("helvetica", 13)
PALETTE_NAMES = {"ironbow": "Ironbow", "white-hot": "White hot", "black-hot": "Black hot"}
WAITING, DISCONNECTED = "Waiting for the camera…", "Camera disconnected"

type Connect = Callable[[OnFrame], Source | None]


def ppm(img: Image) -> bytes:
    h, w, _ = img.shape
    return f"P6 {w} {h} 255 ".encode() + img.tobytes()


class Shot(NamedTuple):
    """One processed frame: raw bytes (photo .bin, phase 3), its numbers and its picture as PPM."""

    raw: bytes
    readouts: Readouts
    picture: bytes


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

        root.title(TITLE)
        root.configure(bg=BG)
        root.resizable(False, False)
        self.photo = tk.PhotoImage(master=root, width=WIDTH, height=HEIGHT)
        self.picture = tk.Label(root, image=self.photo, bg=BG, bd=0, highlightthickness=0)
        self.picture.pack()
        self.message = tk.Label(root, text=WAITING, font=("helvetica", 20), fg=FG, bg=BG, padx=16, pady=8)
        self.message.place(in_=self.picture, relx=0.5, rely=0.5, anchor="center")
        self.debug = tk.Label(root, font=("courier", 12), fg=FG, bg=BG) if debug else None
        if self.debug:
            self.debug.place(in_=self.picture, x=0, y=0)
        self.panel_photo = tk.PhotoImage(master=root, data=ppm(readout_panel(None)))
        tk.Label(root, image=self.panel_photo, bg=BG, bd=0, highlightthickness=0).pack()

        def button(row: tk.Frame, text: str, cmd: Callable[[], None], enabled: bool = True) -> tk.Button:
            b = tk.Button(
                row,
                text=text,
                command=cmd,
                font=UI_FONT,
                fg=FG,
                bg=BUTTON_BG,
                activeforeground=FG,
                activebackground="#333333",
                disabledforeground=DIM,
                relief="flat",
                highlightthickness=0,
                takefocus=False,
                state="normal" if enabled else "disabled",
            )
            b.pack(side="left", expand=True, fill="x", padx=2, pady=2)
            return b

        rows = [tk.Frame(root, bg=BG) for _ in range(2)]
        for row in rows:
            row.pack(fill="x")
        # Photo and Record arrive in phase 3.
        button(rows[0], "Photo", lambda: None, enabled=False)
        button(rows[0], "Record", lambda: None, enabled=False)
        self.palette_button = button(rows[1], "", lambda: self.act("palette"))
        self.lock_button = button(rows[1], "", lambda: self.act("lock"))
        self.rotate_button = button(rows[1], "", lambda: self.act("rotate"))
        self.mirror_button = button(rows[1], "", lambda: self.act("mirror"))
        slider = tk.Frame(root, bg=BG)
        slider.pack(fill="x", padx=8, pady=4)
        tk.Label(slider, text="Smoothing", font=UI_FONT, fg=FG, bg=BG).pack(side="left")
        self.smoothing = tk.DoubleVar(root, settings.smoothing)
        self.scale = tk.Scale(
            slider,
            variable=self.smoothing,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient="horizontal",
            showvalue=False,
            command=self._slid,
            bg="#bbbbbb",  # the slider handle
            troughcolor="#444444",
            activebackground=FG,
            highlightthickness=0,
            bd=0,
            takefocus=False,
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.scale.bind("<ButtonRelease-1>", lambda e: self._released())
        self._label_controls()

        root.bind("<KeyPress>", lambda e: self.act(cfg.key_action(e.keysym) or ""))
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.after(0, self._tick)

    # Reader thread: every frame goes through the tracker, only the latest picture reaches the screen.
    def on_frame(self, raw: bytes, t: Plane) -> None:
        s = self.settings
        shown, r = self.proc.process(t, s.smoothing)
        self.mailbox.put(Shot(raw, r, ppm(compose(shown, r, s.palette, s.smoothing, s.rotation, s.mirror))))

    def act(self, action: str) -> None:
        if action == "quit":
            self.quit()
        elif action == "lock":
            if self.proc.fixed is None:
                self.proc.locked = not self.proc.locked
                self._label_controls()
        else:
            self._set(cfg.apply(self.settings, action))

    def _slid(self, value: str) -> None:
        """Applied live; a drag runs this every 0.01, so the file is written on release (_flush)."""
        if float(value) != self.settings.smoothing:
            self._set(dataclasses.replace(self.settings, smoothing=float(value)), save=False)

    def _released(self) -> None:
        # -command runs at idle: on a fast drag the last value is not applied yet when the release comes.
        self._slid(str(self.scale.get()))
        self._flush()

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
        if self.smoothing.get() != s.smoothing:
            self.smoothing.set(s.smoothing)

    def drop_source(self) -> None:
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
            self.photo.put(shot.picture)
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
        if self.debug and now - self._fps_at >= DEBUG_PERIOD:
            frames = self.source.frames if self.source else 0
            rejected = self.source.rejected if self.source else 0
            fps = (frames - self._fps_frames) / (now - self._fps_at)
            self.debug.configure(text=f"{fps:.1f} fps, rejected: {rejected}, ui {self._ui_ms:.1f} ms")
            self._fps_at, self._fps_frames = now, frames
        self.root.after(TICK_MS, self._tick)

    def _show_readouts(self, shot: Shot | None) -> None:
        r = shot.readouts if shot else None
        self.readout_shot, self.readout_texts = shot, readout_texts(r) if r else None
        self.panel_photo.put(ppm(readout_panel(r)))

    def request_quit(self) -> None:
        """Safe from a signal handler: the next tick quits."""
        self._quit = True

    def quit(self) -> None:
        self._flush()
        self.drop_source()
        self.root.destroy()


def run(
    connect: Connect,
    settings: cfg.Settings,
    settings_path: Path,
    fixed: tuple[float, float] | None = None,
    debug: bool = False,
) -> None:
    """Open the window and run until it is closed; FontError propagates, a missing display exits."""
    for size in (LABEL_SIZE, PAIR_SIZE, CENTRE_SIZE):
        font(size)  # rasterised once here, before the reader thread composes pictures
    try:
        root = tk.Tk(className="tc001-mini-linux")
    except tk.TclError as e:
        sys.exit(f"tc001-mini-linux: cannot open the window: {e}")
    app = App(root, connect, settings, settings_path, fixed, debug)

    def quit_soon(*_: object) -> None:
        app.request_quit()

    previous = {sig: signal.signal(sig, quit_soon) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        root.mainloop()
    finally:
        app.drop_source()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
