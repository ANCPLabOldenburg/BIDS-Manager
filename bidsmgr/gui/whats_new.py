"""What is new: shown once, for the releases a user has not had yet.

Each piece of news (``News``) tells what one or more releases brought:

- **1.5.0**: new features, a redesigned GUI and viewers, a new documentation
  website, beside an animation of the seven kinds of data BIDS Manager reads
  gathering into its logo (``GatherCanvas``);
- **1.5.1 and 1.5.2**: a list of their features beside a stage that plays a
  short painted scene of each (``FeatureStage`` over ``gui/news_scenes.py``).

Which news a start shows depends on the version the user comes from
(``previous_version``): the version that last ran, recorded at every start
(``ui/last_version``); before that record existed, the news last shown
(``ui/whats_new_seen``: "1.5" was shown by 1.5.0 and 1.5.1, so it means 1.5.0
at the latest); else any setting at all, which means an earlier release, from
before What's new existed; else nothing, a first installation. ``due`` lists
every news describing a release newer than that, newest first; a first
installation gets the newest alone. The dialog opens on the newest, and Next
walks to the older ones that are due; within the 1.5.2 news, a release the user
already had is left out of the list. Every news is in the dialog's release
menu and in Home's What's new menu, as Home's sample datasets are.

Called from :func:`bidsmgr.main.main` after the window is up, never from the
window itself, so a test that builds a window gets no dialog. The animations
are painted, run only while visible, and take their colours from the theme
and their fonts from ``gui/viz/fonts``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import QElapsedTimer, QPointF, QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .news_scenes import ALSO, H, SCENES, W, Scene, paint_scene

DOCS_URL = "https://ancplaboldenburg.github.io/bids_manager_documentation/"

_LOGO = Path(__file__).parent / "assets" / "logo.png"

#: The seven kinds of data, in the order they stand around the logo.
KINDS = ("MRI", "MR spectroscopy", "PET", "EEG", "MEG", "Physiology", "Eye tracking")

#: The gathering's timeline, in seconds: the figures appear, hold, gather
#: into the logo (each particle a little after the last, so the stream reads
#: as one), and the logo then breathes.
APPEAR = 0.9
GATHER_AT = 1.6
GATHER = 1.5
STAGGER = 0.6
SETTLED = GATHER_AT + GATHER + STAGGER

#: Seconds a scene takes to appear and to fade at its end.
FADE = 0.35


# ---------------------------------------------------------------------------
# The news, and which a start shows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class News:
    """What one or more releases brought, as one page of the dialog."""

    #: The newest release it tells about.
    key: str
    #: How the menus name it.
    label: str
    #: The releases it tells about, oldest first.
    releases: tuple[str, ...]
    #: Its card on the documentation's Updates page.
    anchor: str

    @property
    def notes_url(self) -> str:
        return DOCS_URL + "updates/#" + self.anchor


#: Every news, newest first.
NEWS: tuple[News, ...] = (
    News("1.5.2", "1.5.1 and 1.5.2", ("1.5.1", "1.5.2"), "v152"),
    News("1.5.0", "1.5.0", ("1.5.0",), "v150"),
)


def version_tuple(text: str) -> tuple[int, int, int]:
    """``"1.5.2"`` -> ``(1, 5, 2)``; a missing part is 0, a suffix
    (``1.5.2.dev0``, ``1.5.2+local``) is ignored."""
    parts = [int(p) for p in re.findall(r"\d+", (text or "").split("+")[0])[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def previous_version(*, last: str, seen: str, used_before: bool) -> Optional[str]:
    """The version the user comes from, or ``None`` for a first installation.

    ``last`` is the version that last ran (recorded since 1.5.2), ``seen``
    the news last shown ("1.5": 1.5.0 or 1.5.1, so 1.5.0 at the latest), and
    ``used_before`` whether any setting exists (a release from before What's
    new: 1.4 or earlier).
    """
    if (last or "").strip():
        return last.strip()
    seen = (seen or "").strip()
    if seen:
        return "1.5.0" if seen == "1.5" else seen
    return "1.4" if used_before else None


def read_previous() -> Optional[str]:
    """``previous_version`` from this computer's settings. Read at start-up,
    before anything of this session is written."""
    from .app_settings import AppSettings

    return previous_version(last=AppSettings.last_version(), seen=AppSettings.whats_new_seen(),
                            used_before=AppSettings.used_before())


def due(previous: Optional[str]) -> list[News]:
    """The news for a user coming from ``previous``, newest first: every one
    telling about a release newer than it. A first installation gets the
    newest alone."""
    if previous is None:
        return [NEWS[0]]
    after = version_tuple(previous)
    return [n for n in NEWS if version_tuple(n.key) > after]


def releases_new_to(news: News, previous: Optional[str]) -> tuple[str, ...]:
    """The releases of ``news`` the user has not had (all of them for a
    first installation, or when the dialog is opened by hand)."""
    if previous is None:
        return news.releases
    after = version_tuple(previous)
    return tuple(r for r in news.releases if version_tuple(r) > after) or news.releases


def maybe_show(window: QWidget, previous: Optional[str] = "") -> Optional["WhatsNewDialog"]:
    """Open the dialog over ``window`` with the news due for a user coming
    from ``previous`` (read from the settings when not given), and record the
    version running now, so the next start shows nothing again.

    Recorded as soon as it opens, so a crash or a closed laptop does not
    bring it back on every start.
    """
    from .. import __version__
    from .app_settings import AppSettings

    if previous == "":
        previous = read_previous()
    news = due(previous)
    AppSettings.remember_last_version(__version__)
    AppSettings.remember_whats_new_seen(NEWS[0].key)
    if not news:
        return None
    return show_dialog(window, news=news, previous=previous, walk=True)


def show_dialog(parent: QWidget, *, news: Optional[Sequence[News]] = None,
                previous: Optional[str] = None, start: Optional[str] = None,
                walk: bool = False) -> "WhatsNewDialog":
    """Open the dialog over ``parent``: the first launch of a release, or
    Home's What's new menu (``start`` names the news to open on). It is
    deleted when closed, so opening it again adds none."""
    dialog = WhatsNewDialog(parent, news=news, previous=previous, start=start, walk=walk)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.open()
    return dialog


# ---------------------------------------------------------------------------
# The particles of the 1.5.0 news: where each one is at time t
# (``logo_points``, ``ease`` and ``seven_colours`` are the star request's too)
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


# ---------------------------------------------------------------------------
# The 1.5.1 and 1.5.2 news: a scene per feature
# ---------------------------------------------------------------------------


def scene_alpha(t: float, duration: float) -> float:
    """How visible a scene is ``t`` seconds in: it appears, holds, and fades
    at its end, so one scene hands over to the next."""
    return max(0.0, min(1.0, t / FADE, (duration - t) / FADE))


class FeatureStage(QWidget):
    """The scenes, one after the other. A scene the user picked plays again
    until another is picked; until then each hands over to the next."""

    scene_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None,
                 scenes: Sequence[Scene] = SCENES) -> None:
        super().__init__(parent)
        from .viz import fonts
        from .viz.bridge import ThemeHub, connect_while_alive

        self.setObjectName("whatsnew-stage")
        self.setMinimumWidth(fonts.px(360))
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.scenes: tuple[Scene, ...] = tuple(scenes)
        self._index = 0
        self._auto = True
        self._clock = QElapsedTimer()
        self._frozen: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._paint_ms = 0.0           # a running average of the paint's cost
        self._describe()
        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w.update())

    # -- which scene, and when --------------------------------------------

    def index(self) -> int:
        return self._index

    def scene(self) -> Scene:
        return self.scenes[self._index]

    def is_auto(self) -> bool:
        """Whether the scenes still hand over to each other by themselves."""
        return self._auto

    def set_scene(self, index: int, *, by_user: bool = False) -> None:
        """Play scene ``index`` from its start; one the user picked plays
        again until another is picked."""
        self._index = index % len(self.scenes)
        if by_user:
            self._auto = False
        if self._clock.isValid():
            self._clock.restart()
        self._describe()
        self.scene_changed.emit(self._index)
        self.update()

    def set_time(self, t: Optional[float]) -> None:
        """Freeze the scene at ``t`` seconds (``None`` lets it run), shown as it
        is at that moment, fading included."""
        self._frozen = t
        self.update()

    def time(self) -> float:
        if self._frozen is not None:
            return self._frozen
        return self._clock.elapsed() / 1000.0 if self._clock.isValid() else 0.0

    def is_running(self) -> bool:
        return self._timer.isActive()

    def _tick(self) -> None:
        if self._frozen is None and self.time() >= self.scene().duration:
            if self._auto:
                self.set_scene(self._index + 1)
            else:
                self._clock.restart()
        self.update()

    def _describe(self) -> None:
        scene = self.scene()
        self.setAccessibleName(f"Animation: {scene.title}")
        self.setAccessibleDescription(scene.shows)

    # -- Qt -----------------------------------------------------------------

    @staticmethod
    def _pads() -> tuple[int, int]:
        """The card's padding around the scene, and the room under it for
        the progress line."""
        from .viz import fonts

        return fonts.px(10), fonts.px(4)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt
        """As tall as the scenes' own proportions (16 by 10) ask."""
        pad, below = self._pads()
        return round((width - 2 * pad) * H / W) + 2 * pad + below

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt
        from .viz import fonts

        width = fonts.px(600)
        return QSize(width, self.heightForWidth(width))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt
        super().showEvent(event)
        if not self._clock.isValid():
            self._clock.start()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt
        from .theme_manager import CUR
        from .viz import fonts

        clock = QElapsedTimer()
        clock.start()
        palette = CUR()
        scene = self.scene()
        t = self.time()
        if self._frozen is None and self._paint_ms > 12.0 and self._timer.interval() == 16:
            self._timer.setInterval(33)     # a slow machine draws half as often

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        card = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = fonts.px(10)
        p.setPen(QPen(QColor(palette.get("border", "#444444")), 1.0))
        p.setBrush(QColor(palette.get("surface", "#222222")))
        p.drawRoundedRect(card, radius, radius)
        p.setBrush(Qt.BrushStyle.NoBrush)

        pad, below = self._pads()
        stage = card.adjusted(pad, pad, -pad, -pad - below)
        p.save()
        p.setClipRect(card.adjusted(1, 1, -1, -1))
        p.setOpacity(scene_alpha(t, scene.duration))
        paint_scene(p, stage, scene, t)
        p.restore()

        # How far the scene has played, along the card's bottom edge.
        done = max(0.0, min(1.0, t / scene.duration))
        y = card.bottom() - fonts.px(6)
        left, right = card.left() + radius, card.right() - radius
        track = QColor(palette.get("subtle", "#333333"))
        p.setPen(QPen(track, max(2.0, fonts.px(2)), Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(int(left), int(y), int(right), int(y))
        if done > 0:
            p.setPen(QPen(QColor(palette.get("dim", "#999999")), max(2.0, fonts.px(2)),
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(int(left), int(y), int(left + (right - left) * done), int(y))
        p.end()
        self._paint_ms = 0.9 * self._paint_ms + 0.1 * clock.nsecsElapsed() / 1e6


# ---------------------------------------------------------------------------
# The dialog: one page per news
# ---------------------------------------------------------------------------


def _text_column(title: str, lede: str) -> tuple[QVBoxLayout, QLabel, QLabel]:
    """A page's right-hand column, started: its title and its lede."""
    from .viz import fonts

    col = QVBoxLayout()
    col.setSpacing(fonts.px(2))
    head = QLabel(title)
    head.setObjectName("whatsnew-title")
    words = QLabel(lede)
    words.setObjectName("whatsnew-lede")
    words.setWordWrap(True)
    col.addWidget(head)
    col.addWidget(words)
    return col, head, words


class ScenesPage(QWidget):
    """The 1.5.1 and 1.5.2 news: the stage and its caption beside the list
    of features, grouped by release, only the releases ``releases``."""

    def __init__(self, releases: Sequence[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .viz import fonts

        scenes = [s for s in SCENES if s.version in releases]
        body = QHBoxLayout(self)
        body.setContentsMargins(fonts.px(18), fonts.px(12), fonts.px(22), fonts.px(10))
        body.setSpacing(fonts.px(22))

        show = QVBoxLayout()
        show.setSpacing(fonts.px(4))
        self.stage = FeatureStage(self, scenes)
        show.addWidget(self.stage)
        show.addSpacing(fonts.px(8))
        self.scene_title = QLabel(self)
        self.scene_title.setObjectName("whatsnew-scene-title")
        self.scene_text = QLabel(self)
        self.scene_text.setObjectName("whatsnew-scene-text")
        self.scene_text.setWordWrap(True)
        self.scene_text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # Room for three lines, so the stage does not jump as the text changes.
        self.scene_text.setMinimumHeight(round(fonts.px(12) * 1.45 * 3))
        show.addWidget(self.scene_title)
        show.addWidget(self.scene_text)
        show.addStretch(1)
        body.addLayout(show, 17)

        both = len(set(releases)) > 1
        side, _title, _lede = _text_column(
            f"BIDS Manager {max(releases, key=version_tuple)}",
            ("What versions " + " and ".join(sorted(set(releases), key=version_tuple))
             + " bring, together." if both else f"What version {releases[0]} brings.")
            + " Pick one to see it again.")
        side.addSpacing(fonts.px(8))
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.feature_buttons: list[QPushButton] = []
        version = None
        for i, scene in enumerate(self.stage.scenes):
            if scene.version != version:
                version = scene.version
                heading = QLabel(f"New in {version}")
                heading.setObjectName("whatsnew-group")
                side.addSpacing(fonts.px(6))
                side.addWidget(heading)
            button = QPushButton(scene.title, self)
            button.setObjectName("whatsnew-feature")
            button.setCheckable(True)
            button.setAutoDefault(False)
            button.setToolTip(scene.text)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _c=False, k=i: self.stage.set_scene(k, by_user=True))
            self._group.addButton(button, i)
            self.feature_buttons.append(button)
            side.addWidget(button)
        side.addSpacing(fonts.px(10))
        also = QLabel(ALSO)
        also.setObjectName("whatsnew-item")
        also.setWordWrap(True)
        side.addWidget(also)
        side.addStretch(1)
        body.addLayout(side, 9)

        self.stage.scene_changed.connect(self._show_scene)
        self._show_scene(self.stage.index())

    def _show_scene(self, index: int) -> None:
        scene = self.stage.scenes[index]
        self.feature_buttons[index].setChecked(True)
        self.scene_title.setText(scene.title)
        self.scene_text.setText(scene.text)

    def step(self, by: int) -> None:
        """The previous or next feature, as the user picks it."""
        self.stage.set_scene(self.stage.index() + by, by_user=True)


class GatherPage(QWidget):
    """The 1.5.0 news: the seven kinds of data gathering into the logo,
    beside what the release brought."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .viz import fonts

        body = QHBoxLayout(self)
        body.setContentsMargins(fonts.px(18), fonts.px(12), fonts.px(28), fonts.px(10))
        body.setSpacing(fonts.px(20))
        self.canvas = GatherCanvas(self)
        body.addWidget(self.canvas, 11)

        text, _title, _lede = _text_column(
            "BIDS Manager 1.5.0",
            "This version brings new features, a redesigned GUI and viewers, and a "
            "new documentation website.")
        text.insertStretch(0, 1)
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


class WhatsNewDialog(QDialog):
    """The news of one or more releases, one page each, chosen in a menu.

    ``news``: the pages Next walks through (all of them by default);
    ``previous``: the version the user comes from, which leaves the releases
    they already had out of a page; ``walk``: opened at start-up, so the main
    button walks to the next page that is due before it reads Continue.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 news: Optional[Sequence[News]] = None, previous: Optional[str] = None,
                 start: Optional[str] = None, walk: bool = False) -> None:
        super().__init__(parent)
        from .viz import fonts
        from .viz.menus import popup_menu

        self.setObjectName("whatsnew-dialog")
        self.setWindowTitle("What's new in BIDS Manager")
        self.setModal(True)
        width, height = fonts.px(1000), fonts.px(620)
        screen = (parent.screen() if parent is not None else None) or self.screen()
        if screen is not None:
            room = screen.availableGeometry()
            width = min(width, int(room.width() * 0.94))
            height = min(height, int(room.height() * 0.9))
        self.resize(width, height)
        self._walk = list(news) if news else list(NEWS)
        self._previous = previous if walk else None
        self._walking = walk and len(self._walk) > 1

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # The kicker, and the menu of every news: a release opens its page.
        top = QHBoxLayout()
        top.setContentsMargins(fonts.px(18), fonts.px(16), fonts.px(22), 0)
        kicker = QLabel("What's new")
        kicker.setObjectName("whatsnew-kicker")
        top.addWidget(kicker)
        top.addStretch(1)
        self.release_button = QPushButton(self)
        self.release_button.setObjectName("tb-btn")
        self.release_button.setToolTip("What each release brought: pick one")
        self.release_menu = popup_menu(self.release_button)
        self._actions = {}
        for item in NEWS:
            action = self.release_menu.addAction(f"What's new in {item.label}")
            action.setCheckable(True)
            action.triggered.connect(lambda _c=False, k=item.key: self.show_news(k))
            self._actions[item.key] = action
        self.release_button.setMenu(self.release_menu)
        top.addWidget(self.release_button)
        outer.addLayout(top)

        self.pages = QStackedWidget(self)
        self._page: dict[str, QWidget] = {}
        outer.addWidget(self.pages, 1)

        footer = QFrame(self)
        footer.setObjectName("whatsnew-footer")
        row = QHBoxLayout(footer)
        row.setContentsMargins(fonts.px(16), fonts.px(10), fonts.px(16), fonts.px(10))
        notes = QPushButton("Release notes", footer)
        notes.setAutoDefault(False)
        notes.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.news().notes_url)))
        star = QPushButton("Give us a star", footer)
        star.setAutoDefault(False)
        star.setToolTip("Why a star on GitHub helps, and how to give one")
        star.clicked.connect(self._show_star_request)
        docs = QPushButton("Open the documentation", footer)
        docs.setAutoDefault(False)
        docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        main = QPushButton("Continue", footer)
        main.setDefault(True)
        main.clicked.connect(self._on_main)
        row.addWidget(notes)
        row.addWidget(star)
        row.addStretch(1)
        row.addWidget(docs)
        row.addWidget(main)
        outer.addWidget(footer)
        self._buttons = {"notes": notes, "star": star, "docs": docs, "continue": main}

        self._key = ""
        self.show_news(start or self._walk[0].key)
        main.setFocus()

    # -- which page ---------------------------------------------------------

    def news(self) -> News:
        return next(n for n in NEWS if n.key == self._key)

    def page(self) -> QWidget:
        return self._page[self._key]

    def show_news(self, key: str) -> None:
        """Show the news ``key``, built the first time it is shown."""
        item = next(n for n in NEWS if n.key == key)
        if key not in self._page:
            if key == "1.5.0":
                page = GatherPage(self.pages)
            else:
                page = ScenesPage(releases_new_to(item, self._previous), self.pages)
            self._page[key] = page
            self.pages.addWidget(page)
        self._key = key
        self.pages.setCurrentWidget(self._page[key])
        for k, action in self._actions.items():
            action.setChecked(k == key)
        self.release_button.setText(f"What's new in {item.label}")
        self._buttons["notes"].setToolTip(item.notes_url)
        main = self._buttons["continue"]
        main.setText(self._main_text())
        # The default button is drawn bold by the stylesheet, after its size
        # was measured in the regular weight: room for the bold text.
        from PyQt6.QtGui import QFont, QFontMetrics

        from .viz import fonts

        bold = QFont(main.font())
        bold.setWeight(QFont.Weight.DemiBold)
        main.setMinimumWidth(QFontMetrics(bold).horizontalAdvance(main.text()) + fonts.px(36))

    def _next_due(self) -> Optional[News]:
        """The next page of the walk, while walking."""
        if not self._walking:
            return None
        keys = [n.key for n in self._walk]
        if self._key not in keys:
            return None
        i = keys.index(self._key)
        return self._walk[i + 1] if i + 1 < len(self._walk) else None

    def _main_text(self) -> str:
        following = self._next_due()
        return f"Next: what {following.label} brought" if following else "Continue"

    def _on_main(self) -> None:
        following = self._next_due()
        if following is not None:
            self.show_news(following.key)
        else:
            self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt
        """Left and right (or up and down) pick the previous or next feature."""
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Up: -1,
                Qt.Key.Key_Right: 1, Qt.Key.Key_Down: 1}.get(Qt.Key(event.key()))
        page = self.page()
        if step is None or not isinstance(page, ScenesPage):
            super().keyPressEvent(event)
            return
        page.step(step)

    def _show_star_request(self) -> None:
        """The request for a GitHub star (``gui/star_request.py``), over this."""
        from .star_request import show_dialog

        show_dialog(self)


__all__ = ["DOCS_URL", "FADE", "FeatureStage", "GatherCanvas", "GatherPage", "KINDS", "NEWS",
           "News", "Particles", "ScenesPage", "WhatsNewDialog", "due", "ease", "logo_points",
           "maybe_show", "previous_version", "read_previous", "releases_new_to", "scene_alpha",
           "seven_colours", "show_dialog", "version_tuple"]
