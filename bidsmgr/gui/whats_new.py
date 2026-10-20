"""What is new in this version: shown once, the first time it opens.

A user who updates, or installs for the first time, sees one dialog that says
what the version brings: new features, the redesigned GUI and viewers, and the
new documentation website. Beside the text, an animation: the seven kinds of
data BIDS Manager reads (MRI, MR spectroscopy, PET, EEG, MEG, physiology, eye
tracking) appear as small figures and then gather into the BIDS Manager logo,
each keeping its colour.

Shown once per release whose notes this is (``NEWS_FOR``), not once per
version string: 1.5.1 does not show the 1.5 news again. Called from
:func:`bidsmgr.main.main` after the window is up, never from the window
itself, so a test that builds a window gets no dialog.

The animation is painted (one ``drawPoints`` per colour), runs only while the
dialog is visible, takes its colours from the theme's plot series
(``VizTheme.series``) and its fonts from ``gui/viz/fonts``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: The release this news describes. Bump it with new text when a release
#: deserves a new dialog; a patch release keeps it, so it is not shown again.
NEWS_FOR = "1.5"

DOCS_URL = "https://ancplaboldenburg.github.io/bids_manager_documentation/"
NOTES_URL = DOCS_URL + "updates/#v150"

_LOGO = Path(__file__).parent / "assets" / "logo.png"

#: The seven kinds of data, in the order they stand around the logo.
KINDS = ("MRI", "MR spectroscopy", "PET", "EEG", "MEG", "Physiology", "Eye tracking")

#: The timeline, in seconds: the figures appear, hold, gather into the logo
#: (each particle a little after the last, so the stream reads as one), and
#: the logo then breathes.
APPEAR = 0.9
GATHER_AT = 1.6
GATHER = 1.5
STAGGER = 0.6
SETTLED = GATHER_AT + GATHER + STAGGER


# ---------------------------------------------------------------------------
# When to show it
# ---------------------------------------------------------------------------


def should_show(seen: str) -> bool:
    """Whether the news of ``NEWS_FOR`` has not been shown yet."""
    return (seen or "").strip() != NEWS_FOR


def maybe_show(window: QWidget) -> Optional["WhatsNewDialog"]:
    """Open the dialog over ``window`` if this release's news is unseen.

    Remembered as soon as it opens, so a crash or a closed laptop does not
    bring it back on every start.
    """
    from .app_settings import AppSettings

    if not should_show(AppSettings.whats_new_seen()):
        return None
    AppSettings.remember_whats_new_seen(NEWS_FOR)
    return show_dialog(window)


def show_dialog(parent: QWidget) -> "WhatsNewDialog":
    """Open the dialog over ``parent`` (the first launch, or Home's What's
    new link); it is deleted when closed, so opening it again adds none."""
    dialog = WhatsNewDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.open()
    return dialog


# ---------------------------------------------------------------------------
# The particles: where each one is at time t
# ---------------------------------------------------------------------------


def logo_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points along the edges of the logo (the folder and the brain's
    folds), centred, the longer side 1."""
    img = QImage(str(_LOGO))
    if img.isNull():  # a missing asset still gives a figure, a ring
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([0.45 * np.cos(a), 0.45 * np.sin(a)], axis=1)
    img = img.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    pixels = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w]
    solid = pixels[..., 3] > 128          # ARGB32 is B, G, R, A in memory
    inner = solid.copy()
    inner[1:-1, 1:-1] &= solid[:-2, 1:-1] & solid[2:, 1:-1] & solid[1:-1, :-2] & solid[1:-1, 2:]
    ys, xs = np.nonzero(solid & ~inner)
    pick = rng.choice(len(xs), size=n, replace=len(xs) < n)
    scale = float(max(w, h))
    return np.stack([(xs[pick] - w / 2) / scale, (ys[pick] - h / 2) / scale], axis=1)


def _figure(kind: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A small figure of one kind of data, as ``n`` points in a [-1, 1] box,
    and the phase each point scrolls with (traces move; images do not)."""
    u = rng.random(n)
    if kind == 0:   # MRI: a slice of a brain, two hemispheres
        a, r = 2 * np.pi * rng.random(n), np.sqrt(rng.random(n))
        rim = rng.random(n) < 0.35
        r = np.where(rim, 1.0, r) * (1 - 0.06 * np.abs(np.sin(7 * a)))
        x, y = 0.9 * r * np.cos(a), 0.75 * r * np.sin(a)
        x = x + np.sign(x) * 0.05
    elif kind == 1:  # MR spectroscopy: a spectrum, its peaks
        x = 2 * u - 1
        peaks = ((0.55, 1.0, 0.035), (0.05, 0.65, 0.03), (-0.12, 0.55, 0.03), (-0.5, 0.35, 0.05))
        y = sum(hgt * np.exp(-((x - c) / wd) ** 2) for c, hgt, wd in peaks)
        y = 0.55 - 1.1 * y + 0.02 * rng.standard_normal(n)
    elif kind == 2:  # PET: an uptake disc, brightest in the middle
        a, r = 2 * np.pi * rng.random(n), 0.9 * rng.random(n) ** 0.9
        x, y = r * np.cos(a), r * np.sin(a)
    elif kind == 3:  # EEG: four channels
        lane = rng.integers(0, 4, n)
        x = 2 * u - 1
        y = -0.66 + 0.44 * lane
    elif kind == 4:  # MEG: a sensor helmet, rings of latitude
        ring = rng.integers(0, 4, n)
        a = np.pi * rng.random(n)
        rad = np.array([1.0, 0.82, 0.58, 0.3])[ring]
        x, y = rad * np.cos(a), 0.45 - 0.95 * rad * np.sin(a) * (1 - 0.18 * ring / 3)
    elif kind == 5:  # physiology: a cardiac trace above, breathing below
        lane = rng.integers(0, 2, n)
        x = 2 * u - 1
        y = np.where(lane == 0, -0.35, 0.45)
    else:            # eye tracking: fixations joined by saccades
        fix = np.array([[-0.8, -0.3], [-0.35, -0.55], [0.1, -0.2], [0.55, -0.5],
                        [0.75, 0.25], [0.2, 0.5], [-0.45, 0.35]])
        seg = rng.integers(0, len(fix) - 1, n)
        on_fix = rng.random(n) < 0.4
        t = rng.random(n)[:, None]
        path = fix[seg] * (1 - t) + fix[seg + 1] * t
        jitter = 0.07 * rng.standard_normal((n, 2))
        pts = np.where(on_fix[:, None], fix[seg] + jitter, path)
        x, y = pts[:, 0], pts[:, 1]
    phase = rng.random(n) * 2 * np.pi
    return np.stack([x, y], axis=1), phase


def _trace(kind: int, base: np.ndarray, t: float) -> np.ndarray:
    """The figure at time ``t``: traces scroll, images hold still."""
    x, y = base[:, 0], base[:, 1].copy()
    if kind == 3:
        s = x + 0.35 * t
        y = y + 0.1 * (np.sin(9 * s + 3 * y) + 0.5 * np.sin(23 * s + 7 * y)) / 1.5
    elif kind == 5:
        s = x + 0.3 * t
        beat = np.mod(s * 2.2, 1.0)
        ecg = 0.5 * np.exp(-((beat - 0.5) / 0.03) ** 2) - 0.12 * np.exp(-((beat - 0.56) / 0.03) ** 2)
        breath = 0.16 * np.sin(5 * s)
        y = np.where(y < 0, y - ecg, y + breath)
    return np.stack([x, y], axis=1)


def ease(s: np.ndarray) -> np.ndarray:
    """Ease in and out (cubic), clipped to 0..1."""
    s = np.clip(s, 0.0, 1.0)
    return np.where(s < 0.5, 4 * s ** 3, 1 - (-2 * s + 2) ** 3 / 2)


class Particles:
    """Every particle's figure, colour group and place in the logo.

    Pure numbers (no Qt but the logo's pixels), so the timeline can be
    tested: :meth:`at` gives each particle's position at time ``t`` in a
    unit frame, the logo centred at the origin and the longer side 1.
    """

    def __init__(self, per_kind: int = 200, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        k = len(KINDS)
        self.n = per_kind * k
        self.kind = np.repeat(np.arange(k), per_kind)
        figures, phases = zip(*(_figure(i, per_kind, rng) for i in range(k)))
        self.figure = np.concatenate(figures)
        self.phase = np.concatenate(phases)
        # Where each figure stands: around the logo, on an ellipse.
        angle = -np.pi / 2 + 2 * np.pi * np.arange(k) / k
        self.anchor = np.stack([0.98 * np.cos(angle), 0.78 * np.sin(angle)], axis=1)
        self.size = 0.17
        # Each kind takes the part of the logo that faces it, so the streams
        # are short and the logo ends up in seven bands of colour.
        # The gathered logo is drawn larger than the ring the figures stand on.
        targets = 1.5 * logo_points(self.n, rng)
        order = np.argsort(np.arctan2(targets[:, 1], targets[:, 0]))
        start = np.argmin(np.mod(np.arctan2(targets[order, 1], targets[order, 0]) - angle[0],
                                 2 * np.pi))
        order = np.roll(order, -start)
        self.target = np.empty_like(targets)
        for i in range(k):
            mine = np.nonzero(self.kind == i)[0]
            self.target[mine] = targets[order[i * per_kind:(i + 1) * per_kind]]
        self.delay = STAGGER * (0.55 * self.kind / k + 0.45 * rng.random(self.n))
        self.members = [np.nonzero(self.kind == i)[0].tolist() for i in range(k)]

    def figure_points(self, t: float) -> np.ndarray:
        out = np.empty((self.n, 2))
        for i in range(len(KINDS)):
            mine = self.kind == i
            out[mine] = self.anchor[i] + self.size * _trace(i, self.figure[mine], t)
        return out

    def at(self, t: float) -> np.ndarray:
        start = self.figure_points(min(t, GATHER_AT))
        if t <= GATHER_AT:
            return start
        s = ease((t - GATHER_AT - self.delay) / GATHER)[:, None]
        mid = (start + self.target) / 2
        bend = np.stack([-(self.target - start)[:, 1], (self.target - start)[:, 0]], axis=1)
        ctrl = mid + 0.18 * bend
        pos = (1 - s) ** 2 * start + 2 * (1 - s) * s * ctrl + s ** 2 * self.target
        if t > SETTLED:
            calm = min(1.0, (t - SETTLED) / 1.5)
            breath = 0.0035 * calm * np.stack([np.sin(1.6 * t + self.phase),
                                               np.cos(1.2 * t + self.phase)], axis=1)
            pos = pos + breath
        return pos

    @staticmethod
    def figures_alpha(t: float) -> float:
        """How visible the figures' labels are: in, held, out as they gather."""
        if t < APPEAR:
            return t / APPEAR
        if t < GATHER_AT:
            return 1.0
        return max(0.0, 1.0 - (t - GATHER_AT) / 0.5)


#: Theme colours a kind of data may take, in order of preference: the plot
#: series first (chosen to stay apart, colour-blind readers included), then
#: other named colours. Which seven are distinct differs per theme (the light
#: themes' fifth series is a violet beside their purple), so they are picked.
_CANDIDATES = ("series1", "series2", "series3", "series4", "series5",
               "purple", "teal", "warning", "error", "success", "accent")


def seven_colours(tokens: dict) -> list[str]:
    """Seven colourful, mutually distinct theme colours for the seven kinds.

    Greedy in OKLab: take the preferred candidates in order, but each next one
    is the candidate farthest from those already taken, among the colourful
    ones (a grey says nothing about a kind of data).
    """
    import math

    from .tones import is_hex, oklch

    pool = []
    for name in _CANDIDATES:
        value = tokens.get(name, "")
        if is_hex(value) and value[:7].lower() not in (c.lower() for c in pool):
            _light, chroma, _hue = oklch(value[:7])
            if chroma >= 0.06:
                pool.append(value[:7])

    def lab(colour: str) -> tuple[float, float, float]:
        light, chroma, hue = oklch(colour)
        return light, chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))

    chosen = pool[:1]
    while len(chosen) < len(KINDS) and len(chosen) < len(pool):
        rest = [c for c in pool if c not in chosen]
        chosen.append(max(rest, key=lambda c: min(math.dist(lab(c), lab(x)) for x in chosen)))
    while len(chosen) < len(KINDS):          # a palette with too few colours repeats
        chosen.append(chosen[len(chosen) % max(1, len(pool))] if pool else "#888888")
    return chosen


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class GatherCanvas(QWidget):
    """The animation: the seven kinds of data gathering into the logo."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .viz import fonts

        self.setObjectName("whatsnew-canvas")
        self.setAccessibleName(
            "Animation: MRI, MR spectroscopy, PET, EEG, MEG, physiology and eye "
            "tracking gathering into the BIDS Manager logo")
        self.setMinimumSize(fonts.px(300), fonts.px(260))
        self.particles = Particles()
        self._clock = QElapsedTimer()
        self._frozen: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)
        self._paint_ms = 0.0           # a running average of the paint's cost
        from .viz.bridge import ThemeHub, connect_while_alive

        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w.update())

    def set_time(self, t: Optional[float]) -> None:
        """Freeze the animation at ``t`` seconds (``None`` lets it run)."""
        self._frozen = t
        self.update()

    def time(self) -> float:
        if self._frozen is not None:
            return self._frozen
        return self._clock.elapsed() / 1000.0 if self._clock.isValid() else 0.0

    def is_running(self) -> bool:
        return self._timer.isActive()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt
        super().showEvent(event)
        if not self._clock.isValid():
            self._clock.start()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt
        from ..viz.theme import VizTheme
        from .theme_manager import CUR
        from .viz import fonts

        # The palette the theme manager applied last, read at every paint:
        # right before the main window has published it to the viewers' hub.
        theme = VizTheme.from_palette(CUR())
        clock = QElapsedTimer()
        clock.start()
        t = self.time()
        slow = self._paint_ms > 12.0       # a slow machine draws half as often
        if self._frozen is None and (t > SETTLED + 2 or slow) and self._timer.interval() == 16:
            self._timer.setInterval(33)   # the logo only breathes now
        w, h = self.width(), self.height()
        unit = min(w / 2.3, h / 2.0)       # the ring of figures has radius ~1 unit
        cx, cy = w / 2, h / 2
        pos = self.particles.at(t)
        xs, ys = (cx + unit * pos[:, 0]).tolist(), (cy + unit * pos[:, 1]).tolist()
        points = list(map(QPointF, xs, ys))

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dot = max(1.6, fonts.px(2) * 1.15)
        palette = {**theme.tokens, **{f"series{i + 1}": theme.series(i) for i in range(6)}}
        colours = seven_colours(palette)
        for i in range(len(KINDS)):
            mine = self.particles.members[i]
            colour = QColor(colours[i])
            colour.setAlphaF(0.9)
            pen = QPen(colour, dot, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawPoints([points[j] for j in mine])

        alpha = Particles.figures_alpha(t)
        if alpha > 0:
            p.setFont(fonts.font(fonts.LABEL_PX))
            text = QColor(theme.dim)
            text.setAlphaF(alpha)
            p.setPen(text)
            box_w, box_h = unit * 0.6, fonts.px(16)
            for i, name in enumerate(KINDS):
                ax = cx + unit * self.particles.anchor[i][0]
                ay = cy + unit * (self.particles.anchor[i][1] + self.particles.size * 1.25)
                p.drawText(QRectF(ax - box_w / 2, ay, box_w, box_h),
                           Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, name.upper())
        p.end()
        self._paint_ms = 0.9 * self._paint_ms + 0.1 * clock.nsecsElapsed() / 1e6


class WhatsNewDialog(QDialog):
    """What this release brings, beside the gathering animation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .viz import fonts

        self.setObjectName("whatsnew-dialog")
        self.setWindowTitle("What's new in BIDS Manager")
        self.setModal(True)
        self.resize(fonts.px(920), fonts.px(560))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(fonts.px(16), fonts.px(16), fonts.px(28), fonts.px(8))
        body.setSpacing(fonts.px(20))
        self.canvas = GatherCanvas(self)
        body.addWidget(self.canvas, 11)

        text = QVBoxLayout()
        text.setSpacing(fonts.px(6))
        kicker = QLabel("What's new")
        kicker.setObjectName("whatsnew-kicker")
        title = QLabel(f"BIDS Manager {NEWS_FOR}")
        title.setObjectName("whatsnew-title")
        lede = QLabel(
            "This version brings new features, a redesigned GUI and viewers, and a "
            "new documentation website.")
        lede.setObjectName("whatsnew-lede")
        lede.setWordWrap(True)
        text.addStretch(1)
        for widget in (kicker, title, lede):
            text.addWidget(widget)
        text.addSpacing(fonts.px(10))
        for heading, words in (
            ("New features",
             "MRI series named from their acquisition parameters, not only from the "
             "name typed at the scanner. A preview of the BIDS dataset before you convert, "
             "an overview of your scan, eye tracking, a quick image-quality check, and "
             "a copy of your source data if you want one."),
            ("A redesigned GUI and viewers",
             "New themes and typefaces, panels as cards, and one viewer for every kind "
             "of data: MRI in planes and 3-D with overlays and quality maps, PET, "
             "spectroscopy, EEG, MEG and physiology."),
            ("A new documentation website",
             "Rebuilt from scratch: a step-by-step course on one sample dataset, a "
             "guide for each kind of data, and a reference for every setting and "
             "command."),
        ):
            head = QLabel(heading)
            head.setObjectName("whatsnew-item-title")
            line = QLabel(words)
            line.setObjectName("whatsnew-item")
            line.setWordWrap(True)
            text.addWidget(head)
            text.addWidget(line)
            if heading == "New features":
                text.addLayout(self._credit_row())
            text.addSpacing(fonts.px(6))
        text.addStretch(1)
        body.addLayout(text, 9)
        outer.addLayout(body, 1)

        footer = QFrame(self)
        footer.setObjectName("whatsnew-footer")
        row = QHBoxLayout(footer)
        row.setContentsMargins(fonts.px(16), fonts.px(10), fonts.px(16), fonts.px(10))
        notes = QPushButton("Release notes", footer)
        notes.setAutoDefault(False)
        notes.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(NOTES_URL)))
        star = QPushButton("Give us a star", footer)
        star.setAutoDefault(False)
        star.setToolTip("Why a star on GitHub helps, and how to give one")
        star.clicked.connect(self._show_star_request)
        docs = QPushButton("Open the documentation", footer)
        docs.setAutoDefault(False)
        docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        start = QPushButton("Continue", footer)
        start.setDefault(True)
        start.clicked.connect(self.accept)
        row.addWidget(notes)
        row.addWidget(star)
        row.addStretch(1)
        row.addWidget(docs)
        row.addWidget(start)
        outer.addWidget(footer)
        self._buttons = {"notes": notes, "star": star, "docs": docs, "continue": start}
        start.setFocus()

    def _show_star_request(self) -> None:
        """The request for a GitHub star (``gui/star_request.py``), over this."""
        from .star_request import show_dialog

        show_dialog(self)

    def _credit_row(self) -> QHBoxLayout:
        """The MRI naming is an early implementation, and part of its idea
        comes from the SWIFTCAT team: one line, the full note behind the icon
        (the same note as Settings > Classification)."""
        from .classifier_rules import show_about
        from .viz.panels.panel_header import corner_button

        row = QHBoxLayout()
        self.credit_button = corner_button(
            "info", "How MRI series are named, and credit to the SWIFTCAT team")
        self.credit_button.clicked.connect(lambda: show_about(self.credit_button))
        label = QLabel("MRI naming: an early implementation, and credit to the SWIFTCAT team")
        label.setObjectName("whatsnew-item")
        label.setWordWrap(True)
        row.addWidget(self.credit_button)
        row.addWidget(label, 1)
        return row


__all__ = ["DOCS_URL", "KINDS", "NEWS_FOR", "NOTES_URL", "Particles", "GatherCanvas",
           "WhatsNewDialog", "ease", "logo_points", "maybe_show", "seven_colours", "should_show",
           "show_dialog"]
