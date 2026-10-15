"""Tracks: plots on a shared time axis, one per measure, each its own.

The plots under a BOLD's time course (its QC and the run's physiology) and
under an MEG or EEG recording (its QC) are TRACKS. Each is a small plot of
its own in a rounded card:

* a header ABOVE the plot with the title, the curves' legend (a coloured
  line beside a name in text colour, never coloured text), and the value
  under the pointer; so a title can never cover the signal;
* its own y axis in its own units (mm, %, z), never two scales on one axis;
* the x axis shared with the view above: Ctrl+wheel zooms time, Shift+wheel
  pans it, double-click shows all of it (the host decides, so the time
  course and its tracks move together);
* hover reads every track at the same moment (one cursor across all of
  them), a click goes there;
* the current volume or time as a line, flagged stretches shaded, the
  window on screen outlined;
* move up, move down and hide in the header, so the user orders them.

The panel FITS the tracks in the room it has, or in scroll mode gives each
a readable height in a scrolling column. Everything is in the theme's
colours and follows a theme change.

A track is plain data built on a worker (a dict):

``id`` stable identity; ``title``; ``unit`` ("" for none); ``kind`` "line",
"events", "image" or "pending" (being computed: a spinner and ``summary``,
no plot yet, at the size the plot will have); ``explain`` the key of its
explanation (``bidsmgr.qc.explain``: an info icon opens it); ``x`` positions in the panel's x units; ``ys`` one or
more curves; ``legend`` their names; ``colour`` a channel-type or theme
token for a single curve; ``ticks`` marked positions; ``rule`` a threshold
in the track's units (dashed); ``y_range`` fixed limits; ``image`` (rows x
columns) with ``image_x`` (left, right) and ``levels``, ``rows`` naming its
rows; ``bands`` shaded stretches (x0, x1); ``summary`` what the header says
when the pointer is elsewhere; ``note`` the tooltip; ``height`` relative
height; ``movable`` / ``closable``; ``fmt`` a value format.

pyqtgraph items are POOLED (guard 8d): created once per card, reused and
hidden, never removed.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QObject, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from ....viz.theme import parse_colour
from ...widgets.primitives import ElidedLabel
from ..bridge import connect_while_alive

log = logging.getLogger(__name__)

#: Height of a track per unit of its ``height`` when they share the room
#: (the least it shrinks to) and in scroll mode (what each gets).
FIT_PX = 92
SCROLL_PX = 160
#: Room for the x axis' numbers under the last track.
AXIS_PX = 24


def nice_ticks(lo: float, hi: float, n: int = 3) -> list[float]:
    """About ``n`` round values inside ``[lo, hi]`` (1, 2, 2.5 or 5 times a
    power of ten apart): a small plot labelled at every pyqtgraph tick reads
    as a smudge of digits."""
    import math

    span = hi - lo
    if not (math.isfinite(lo) and math.isfinite(hi)) or span <= 0:
        return [lo] if math.isfinite(lo) else []
    mag = 10.0 ** math.floor(math.log10(span))

    def within(step: float) -> list[float]:
        v = math.ceil(lo / step - 1e-9) * step
        out = []
        while v <= hi + 1e-9 * span and len(out) <= n + 2:
            out.append(0.0 if abs(v) < step * 1e-9 else round(v, 12))
            v += step
        return out

    # The largest round step that still labels two to n + 1 values.
    for step in sorted((m * mag * f for f in (1.0, 0.1, 0.01) for m in (5.0, 2.0, 1.0)),
                       reverse=True):
        got = within(step)
        if 2 <= len(got) <= n + 1:
            return got
    return [round(lo, 12), round(hi, 12)]


def _tick_label(v: float) -> str:
    text = f"{v:.4g}"
    return "0" if text in ("-0", "0") else text


def _qcolor(value: str, alpha: Optional[int] = None) -> QColor:
    c = QColor(*parse_colour(value))
    if alpha is not None:
        c.setAlpha(alpha)
    return c


class _Legend(QWidget):
    """The curves' names, each after a short line in its colour. Painted
    (guard 8e); the names are in the text colour, the line carries identity."""

    def __init__(self) -> None:
        super().__init__()
        self._items: list[tuple[str, str]] = []
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_items(self, items: list[tuple[str, str]]) -> None:
        if items != self._items:
            self._items = list(items)
            self.updateGeometry()
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        fm = self.fontMetrics()
        width = sum(14 + 4 + fm.horizontalAdvance(name) + 10 for name, _c in self._items)
        return QSize(width, fm.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return self.sizeHint()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        if not self._items:
            return
        from ..bridge import ThemeHub

        theme = ThemeHub.instance().theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = self.fontMetrics()
        x = 0.0
        mid = self.height() / 2.0
        for name, colour in self._items:
            p.setPen(QPen(_qcolor(colour), 2.0))
            p.drawLine(int(x), int(mid), int(x + 14), int(mid))
            x += 18
            p.setPen(_qcolor(theme.dim))
            w = fm.horizontalAdvance(name)
            p.drawText(QRectF(x, 0, w + 2, self.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)
            x += w + 10
        p.end()


class TrackCard(QFrame):
    """One track: header and plot in a rounded card."""

    def __init__(self, panel: "TracksPanel", track_id: str) -> None:
        super().__init__()
        import pyqtgraph as pg

        self._pg = pg
        self.panel = panel
        self.track_id = track_id
        self.track: dict = {}
        self.setObjectName("viz-track")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 4, 6, 4)
        v.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(2, 0, 0, 0)
        head.setSpacing(8)
        self.title = QLabel("")
        self.title.setObjectName("viz-track-title")
        head.addWidget(self.title)
        # Being computed: a spinner beside the title until the plot arrives.
        from ...widgets.spinner import BusySpinner

        self.spinner = BusySpinner()
        self.spinner.setVisible(False)
        head.addWidget(self.spinner)
        self.legend = _Legend()
        head.addWidget(self.legend)
        self.readout = ElidedLabel("")
        self.readout.setObjectName("viz-track-readout")
        head.addWidget(self.readout, 1)
        self.info = self._button("info", "What this plot shows and how to read it",
                                 self._explain)
        self.info.setVisible(False)
        head.addWidget(self.info)
        self.up = self._button("chevron_up", "Move this plot up", lambda: panel._move(self, -1))
        self.down = self._button("chevron_down", "Move this plot down",
                                 lambda: panel._move(self, 1))
        self.hide_button = self._button("close", "Hide this plot (Plots brings it back)",
                                        lambda: panel.hide_requested.emit(self.track_id))
        for b in (self.up, self.down, self.hide_button):
            head.addWidget(b)
        v.addLayout(head)

        self.plot = pg.PlotWidget()
        self.plot.setObjectName("viz-track-plot")
        pi = self.plot.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.setMouseEnabled(x=False, y=False)
        self._vb = pi.getViewBox()
        self._vb.disableAutoRange()
        for name in ("left", "bottom"):
            pi.getAxis(name).enableAutoSIPrefix(False)
        pi.getAxis("left").setWidth(panel.left_axis_px)
        pi.layout.setContentsMargins(0, 2, 6, 0)
        self.plot.setMinimumHeight(30)
        self.plot.wheelEvent = self._wheel
        self.plot.leaveEvent = self._leave
        self.plot.mouseDoubleClickEvent = lambda _e: panel.reset_view.emit()
        self.plot.scene().sigMouseMoved.connect(self._moved)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        v.addWidget(self.plot, 1)

        # Pooled items, created once.
        self._bands = pg.BarGraphItem(x0=[0.0], x1=[0.0], y0=[0.0], y1=[0.0])
        self._regions = pg.BarGraphItem(x0=[0.0], x1=[0.0], y0=[0.0], y1=[0.0])
        self._window = pg.BarGraphItem(x0=[0.0], x1=[0.0], y0=[0.0], y1=[0.0])
        for item, z in ((self._bands, -12), (self._regions, -11), (self._window, -10)):
            item.setZValue(z)
            item.setVisible(False)
            pi.addItem(item)
        self._image = None
        self._curves: list = []
        self._ticks = pg.PlotCurveItem(connect="pairs", antialias=False)
        self._rule = pg.PlotCurveItem(connect="pairs", antialias=False)
        self._marker = pg.PlotCurveItem(antialias=False)
        self._hover = pg.PlotCurveItem(antialias=False)
        for item, z in ((self._rule, -5), (self._ticks, 5), (self._hover, 8), (self._marker, 9)):
            item.setZValue(z)
            pi.addItem(item)
        self._y = (0.0, 1.0)

    def _button(self, icon: str, tip: str, slot) -> QPushButton:
        from ... import icons

        b = QPushButton()
        b.setObjectName("viz-track-btn")
        b.setIcon(icons.icon(icon))
        b.setIconSize(QSize(14, 14))
        b.setFixedSize(22, 20)
        b.setToolTip(tip)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(slot)
        b._icon_name = icon
        return b

    def _explain(self) -> None:
        from ..panels import explain_popup

        t = self.track or {}
        explain_popup.show(self.info, t.get("explain", ""), title=t.get("title", ""),
                           here=t.get("note", ""), fallback=t.get("note", ""))

    # -- events ---------------------------------------------------------------

    def _wheel(self, event) -> None:
        # Ignored unless the host takes it (Ctrl/Shift zoom and pan time), so
        # a plain wheel scrolls the column of tracks.
        event.ignore()
        self.panel.wheel.emit(event, self)

    def _leave(self, _event) -> None:
        self.panel._hover_at(None, None)

    def view_x(self, scene_pos) -> Optional[tuple[float, float]]:
        if not self._vb.sceneBoundingRect().contains(scene_pos):
            return None
        p = self._vb.mapSceneToView(scene_pos)
        return float(p.x()), float(p.y())

    def _moved(self, scene_pos) -> None:
        got = self.view_x(scene_pos)
        self.panel._hover_at(got[0] if got else None, self if got else None,
                             got[1] if got else None)

    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        got = self.view_x(event.scenePos())
        if got is None:
            return
        event.accept()
        row = self.row_at(got[0], got[1])
        if row is not None:
            # A cell of an image (a channel map): which row, at what moment.
            self.panel.row_clicked.emit(got[0], row)
        else:
            self.panel.clicked.emit(got[0])

    def row_at(self, x: float, y: Optional[float]) -> Optional[str]:
        """The row of an image track under (``x``, ``y``), by name; None off
        the image or on a track that is not one."""
        t = self.track or {}
        if t.get("kind") != "image" or y is None:
            return None
        image = np.asarray(t["image"])
        if not 0 <= y < image.shape[0]:
            return None
        x0, x1 = t["image_x"]
        if not x0 <= x < x1:
            return None
        r = image.shape[0] - 1 - int(y)
        rows = t.get("rows") or []
        return rows[r] if r < len(rows) else None

    # -- content --------------------------------------------------------------

    def set_track(self, track: dict) -> None:
        self.track = track
        theme = self.panel.theme()
        unit = track.get("unit", "")
        self.title.setText(track["title"] + (f" ({unit})" if unit else ""))
        kind = track.get("kind", "line")
        pending = kind == "pending"
        if self.spinner.is_busy() != pending:           # only on a real change
            self.spinner.set_busy(pending)
        has_info = bool(track.get("explain") or track.get("note"))
        if self.info.isHidden() == has_info:
            self.info.setVisible(has_info)
        if track.get("explain"):
            from ..panels.explain_popup import hover_text

            self.setToolTip(hover_text(track["explain"]))
        else:
            self.setToolTip(track.get("note", ""))
        self.legend.set_items([(name, theme.series(j))
                               for j, name in enumerate(track.get("legend") or [])])
        self.up.setVisible(bool(track.get("movable", True)))
        self.down.setVisible(bool(track.get("movable", True)))
        self.hide_button.setVisible(bool(track.get("closable", True)))
        self.readout.setText(track.get("summary", ""))
        pi = self.plot.getPlotItem()
        ys = list(track.get("ys") or []) if kind == "line" else []
        x = np.asarray(track.get("x", np.empty(0)), dtype=float)
        while len(self._curves) < len(ys):
            curve = self._pg.PlotCurveItem(antialias=True)
            pi.addItem(curve)
            self._curves.append(curve)
        for k, curve in enumerate(self._curves):
            if k < len(ys):
                curve.setData(x, np.asarray(ys[k], dtype=float), connect="finite")
                curve.setVisible(True)
            else:
                curve.setVisible(False)
        lo, hi = self._y_limits(track, ys)
        self._y = (lo, hi)
        self._vb.setYRange(lo, hi, padding=0.0)
        if kind == "line":
            pad = (hi - lo) * 0.08 / 1.16
            values = nice_ticks(lo + pad, hi - pad)
            pi.getAxis("left").setTicks([[(v, _tick_label(v)) for v in values], []])
        # Marks: full height for an events track, a band along the top of a
        # curve (the volumes a threshold flags).
        ticks = np.asarray(track.get("ticks", np.empty(0)), dtype=float)
        if ticks.size:
            top0, top1 = ((lo, hi) if kind == "events"
                          else (hi - 0.14 * (hi - lo), hi - 0.02 * (hi - lo)))
            self._ticks.setData(np.repeat(ticks, 2), np.tile([top0, top1], ticks.size))
        else:
            self._ticks.setData([], [])
        rule = track.get("rule")
        if rule is not None and np.isfinite(rule) and x.size:
            self._rule.setData([float(np.nanmin(x)), float(np.nanmax(x))], [rule, rule])
        else:
            self._rule.setData([], [])
        bands = track.get("bands") or []
        self._set_bars(self._bands, bands)
        self._set_image(track if kind == "image" else None)
        left = pi.getAxis("left")
        left.setStyle(showValues=kind == "line")
        if kind == "pending":
            # Nothing to scale yet: no bare tick marks down an empty plot.
            left.setTicks([[], []])
        self.apply_theme()

    @staticmethod
    def _y_limits(track: dict, ys: list) -> tuple[float, float]:
        if track.get("kind") == "image":
            return 0.0, float(np.asarray(track["image"]).shape[0])
        if track.get("kind") == "events":
            return 0.0, 1.0
        if track.get("y_range") is not None:
            lo, hi = (float(v) for v in track["y_range"])
        else:
            values = [np.asarray(y, dtype=float) for y in ys]
            finite = (np.concatenate([v[np.isfinite(v)] for v in values])
                      if values else np.empty(0))
            if finite.size == 0:
                return 0.0, 1.0
            lo, hi = float(finite.min()), float(finite.max())
            rule = track.get("rule")
            if rule is not None and np.isfinite(rule):
                lo, hi = min(lo, rule), max(hi, rule)
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    def _set_bars(self, item, spans, *, alpha: int = 40, colour: Optional[str] = None,
                  outline: bool = False) -> None:
        if not spans:
            item.setVisible(False)
            return
        lo, hi = self._y
        x0 = [float(a) for a, _b in spans]
        x1 = [float(b) for _a, b in spans]
        theme = self.panel.theme()
        fill = _qcolor(colour or theme.dim, alpha)
        pen = (self._pg.mkPen(_qcolor(colour or theme.accent), width=1)
               if outline else self._pg.mkPen(None))
        item.setOpts(x0=x0, x1=x1, y0=[lo] * len(x0), y1=[hi] * len(x0), brush=fill, pen=pen)
        item.setVisible(True)

    def _set_image(self, track: Optional[dict]) -> None:
        if track is None:
            if self._image is not None:
                self._image.setVisible(False)
            return
        if self._image is None:
            self._image = self._pg.ImageItem()
            self._image.setZValue(-8)
            self.plot.getPlotItem().addItem(self._image)
        image = np.nan_to_num(np.asarray(track["image"], dtype=np.float32))
        lo, hi = track.get("levels", (-2.0, 2.0))
        # Row 0 at the top; pyqtgraph's y runs up.
        self._image.setImage(image[::-1].T, levels=(lo, hi), autoLevels=False)
        self._image.setLookupTable(self._lut(track.get("colormap") or ""))
        x0, x1 = track["image_x"]
        self._image.setRect(QRectF(float(x0), 0.0, float(x1) - float(x0), float(image.shape[0])))
        self._image.setVisible(True)

    def _lut(self, name: str):
        """The look-up table of an image track's colour map: ``diverging``
        is blue, the plot's own background in the middle, then red, so what
        is normal (0) fades into the card in either theme and only what
        departs from it shows; a named map otherwise; grey without one (a
        card can be reused)."""
        if not name:
            return None
        if name == "diverging":
            theme = self.panel.theme()
            mid = np.asarray(parse_colour(theme.plot_background)[:3], dtype=float)
            blue = np.array([59.0, 130.0, 246.0])
            red = np.array([239.0, 68.0, 68.0])
            t = np.linspace(0.0, 1.0, 128)[:, None]
            low = blue + (mid - blue) * t
            high = mid + (red - mid) * t
            return np.clip(np.rint(np.vstack([low, high])), 0, 255).astype(np.uint8)
        from ....viz.compute import colormaps

        return colormaps.lut(name) if colormaps.exists(name) else None

    def set_regions(self, spans) -> None:
        theme = self.panel.theme()
        self._set_bars(self._regions, spans, alpha=60,
                       colour=theme.token("warning", "#d29922"))

    def set_window(self, span) -> None:
        theme = self.panel.theme()
        self._set_bars(self._window, [span] if span else [], alpha=28, colour=theme.accent,
                       outline=True)

    def set_marker(self, x: Optional[float]) -> None:
        lo, hi = self._y
        if x is None:
            self._marker.setData([], [])
        else:
            self._marker.setData([x, x], [lo, hi])

    def set_hover(self, x: Optional[float]) -> None:
        lo, hi = self._y
        if x is None:
            self._hover.setData([], [])
        else:
            self._hover.setData([x, x], [lo, hi])

    def read(self, x: float, y: Optional[float]) -> str:
        """What the track says at ``x`` (and ``y``, for an image)."""
        t = self.track
        kind = t.get("kind", "line")
        fmt = t.get("fmt", "{:.3g}")
        unit = t.get("unit", "")
        if kind == "image":
            image = np.asarray(t["image"])
            rows = t.get("rows") or []
            if y is None or not 0 <= y < image.shape[0]:
                return ""
            r = image.shape[0] - 1 - int(y)
            x0, x1 = t["image_x"]
            c = int((x - x0) / max(x1 - x0, 1e-12) * image.shape[1])
            if not 0 <= c < image.shape[1]:
                return ""
            name = rows[r] if r < len(rows) else f"row {r + 1}"
            return f"{name}: {fmt.format(float(image[r, c]))} {t.get('value_name', '')}".strip()
        if kind == "events":
            ticks = np.asarray(t.get("ticks", np.empty(0)), dtype=float)
            near = ticks.size and float(np.min(np.abs(ticks - x))) <= t.get("near", 0.5)
            return "marked" if near else ""
        xs = np.asarray(t.get("x", np.empty(0)), dtype=float)
        if not xs.size:
            return ""
        i = int(np.clip(np.searchsorted(xs, x), 0, xs.size - 1))
        if i > 0 and abs(xs[i - 1] - x) < abs(xs[i] - x):
            i -= 1
        values = [float(np.asarray(y_)[i]) for y_ in t.get("ys") or []]
        names = t.get("legend") or []
        if not values:
            return ""
        if names:
            body = "  ".join(f"{n} {fmt.format(v)}" for n, v in zip(names, values))
        else:
            body = fmt.format(values[0])
        return f"{body} {unit}".strip()

    # -- look ---------------------------------------------------------------

    def apply_theme(self) -> None:
        pg = self._pg
        theme = self.panel.theme()
        self.plot.setBackground(theme.plot_background)
        pi = self.plot.getPlotItem()
        for name in ("left", "bottom"):
            pi.getAxis(name).setPen(_qcolor(theme.plot_foreground, 120))
        from .. import fonts

        fonts.style_axes(pi, _qcolor(theme.plot_foreground, 200).name())
        t = self.track
        colour = t.get("colour") or "accent"
        # A theme token (accent, purple, teal, warning, dim) or a channel type.
        single = (theme.accent if colour == "accent" else
                  theme.tokens[colour] if colour in theme.tokens else theme.type_colour(colour))
        for k, curve in enumerate(self._curves):
            pen = theme.series(k) if t.get("legend") else single
            curve.setPen(pg.mkPen(_qcolor(pen), width=1.5))
        warn = theme.token("warning", "#d29922")
        self._ticks.setPen(pg.mkPen(_qcolor(warn), width=1.5))
        self._rule.setPen(pg.mkPen(_qcolor(warn, 200), width=1, style=Qt.PenStyle.DashLine))
        self._marker.setPen(pg.mkPen(_qcolor(self.panel.ctx.settings.crosshair.color), width=1.5))
        self._hover.setPen(pg.mkPen(_qcolor(theme.dim, 170), width=1))
        if t.get("kind") == "image" and self._image is not None:
            # A theme-dependent map follows the theme.
            self._image.setLookupTable(self._lut(t.get("colormap") or ""))
        if self._bands.isVisible():
            self._set_bars(self._bands, self.track.get("bands") or [])
        for b in (self.up, self.down, self.hide_button):
            from ... import icons

            b.setIcon(icons.icon(b._icon_name))
        self.legend.update()

    def align(self, left: int, right: int) -> None:
        """Put the plot area at ``left`` .. ``right`` (global x), where the
        view above has its own, so a moment is at the same pixel in both:
        the left axis takes the room before it, a right margin the room
        after it (the plot area starts at margin + axis width, measured)."""
        origin = self.plot.mapToGlobal(self.plot.rect().topLeft()).x()
        axis = max(28, int(left - origin))
        margin = max(0, int(origin + self.plot.width() - right))
        pi = self.plot.getPlotItem()
        ax = pi.getAxis("left")
        if int(ax.width()) != axis:
            ax.setWidth(axis)
        if getattr(self, "_right_margin", None) != margin:
            self._right_margin = margin
            pi.layout.setContentsMargins(0, 2, margin, 0)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        # A scroll bar appearing narrows the cards, not the panel.
        self.panel.request_align()

    def show_x_axis(self, on: bool, label: str) -> None:
        """Only the last track carries the time axis: a ticked line under
        every plot reads as noise."""
        pi = self.plot.getPlotItem()
        if on:
            pi.showAxis("bottom")
            bottom = pi.getAxis("bottom")
            bottom.setHeight(AXIS_PX)
            # Numbers only: the view above names the axis, and in a track's
            # height a title collided with them.
            pi.setLabel("bottom", None)
            bottom.setToolTip(label)
        else:
            pi.hideAxis("bottom")


class _ResizeWatch(QObject):
    """Calls back when a watched widget is resized (an event filter, owned
    by the watched widget so it goes with it)."""

    def __init__(self, target: QWidget, callback: Callable[[], None]) -> None:
        super().__init__(target)
        self._callback = callback
        target.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt naming
        from PyQt6.QtCore import QEvent

        if event.type() == QEvent.Type.Resize:
            self._callback()
        return False


class TracksPanel(QWidget):
    """Tracks one above the other, fitted or scrolling."""

    #: A click on a track at x (go there).
    clicked = pyqtSignal(float)
    #: A click on a named row of an image track (a channel map): x and the
    #: row's name (go to that channel there).
    row_clicked = pyqtSignal(float, str)
    #: A wheel event over a track (the host zooms or pans time, or ignores it).
    wheel = pyqtSignal(object, object)
    #: Double-click: show everything again.
    reset_view = pyqtSignal()
    #: The user moved a track: its id and the new order of ids.
    order_changed = pyqtSignal(list)
    #: The user hid a track.
    hide_requested = pyqtSignal(str)

    def __init__(self, ctx, *, left_axis_px: int = 56, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.left_axis_px = int(left_axis_px)
        self.setObjectName("viz-tracks")
        self._cards: dict[str, TrackCard] = {}
        self._order: list[str] = []
        self._mode = "fit"
        self._x_label = ""
        self._x_range: Optional[tuple[float, float]] = None
        self._marker: Optional[float] = None
        self._regions: list = []
        self._window = None
        #: How the readouts name the moment under the pointer.
        self.describe_x: Callable[[float], str] = lambda x: f"{x:g}"
        #: Where the view above has its plot area, as global (left, right),
        #: or None: the tracks line their own up with it.
        self.align_source: Optional[Callable[[], Optional[tuple[int, int]]]] = None
        from PyQt6.QtCore import QTimer

        self._align_timer = QTimer(self)
        self._align_timer.setSingleShot(True)
        self._align_timer.setInterval(0)
        self._align_timer.timeout.connect(self.align)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("viz-tracks-scroll")
        self.scroll.viewport().setObjectName("viz-tracks-viewport")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body.setObjectName("viz-tracks-body")
        self._lay = QVBoxLayout(self.body)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self._lay.setSpacing(6)
        self._lay.addStretch(0)
        self.scroll.setWidget(self.body)
        outer.addWidget(self.scroll)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.apply_theme())

    def theme(self):
        return self.ctx.theme

    # -- content ---------------------------------------------------------------

    def set_tracks(self, tracks: list[dict], *, x_label: str = "") -> None:
        """Show ``tracks``, in this order (cards are kept per id and reused)."""
        self._x_label = x_label
        ids = [t["id"] for t in tracks]
        for t in tracks:
            card = self._cards.get(t["id"])
            if card is None:
                card = TrackCard(self, t["id"])
                self._cards[t["id"]] = card
            card.set_track(t)
        if ids != self._order:
            for tid in self._order:
                card = self._cards.get(tid)
                if card is not None:
                    self._lay.removeWidget(card)
                    if tid not in ids:
                        card.setVisible(False)
            for k, tid in enumerate(ids):
                self._lay.insertWidget(k, self._cards[tid])
                self._cards[tid].setVisible(True)
            self._order = ids
        self._apply_mode()
        for card in self.cards():
            if self._x_range is not None:
                card.plot.setXRange(*self._x_range, padding=0.0)
            card.set_regions(self._regions)
            card.set_window(self._window)
            card.set_marker(self._marker)
            card.up.setEnabled(card.track_id != ids[0] if ids else False)
            card.down.setEnabled(card.track_id != ids[-1] if ids else False)
        self.request_align()

    def reach(self) -> Optional[tuple[int, int]]:
        """How far the plot areas CAN go, as global (left, right): the
        leftmost start that keeps a readable axis, the rightmost end the
        narrowest card allows. The view above is fitted to the same span."""
        cards = [c for c in self.cards() if c.isVisible()]
        if not cards:
            return None
        lefts, rights = [], []
        for c in cards:
            origin = c.plot.mapToGlobal(c.plot.rect().topLeft()).x()
            lefts.append(origin + 28)
            rights.append(origin + c.plot.width())
        return max(lefts), min(rights)

    def align(self) -> None:
        """Line every plot area up with the view above (see ``align_source``,
        which may also move the view's own edge to meet the tracks)."""
        got = self.align_source() if self.align_source is not None else None
        if got is None:
            return
        for card in self.cards():
            card.align(*got)

    def request_align(self) -> None:
        """Align once the layout has settled (coalesced)."""
        self._align_timer.start()

    def follow(self, widget: QWidget) -> None:
        """Re-align whenever ``widget`` (the view above) is resized: it can
        change size after the tracks do (a controls column opening)."""
        if getattr(widget, "_tracks_watch", None) is None:
            widget._tracks_watch = _ResizeWatch(widget, self.request_align)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self.request_align()

    def can_scroll(self) -> bool:
        """Whether the column of tracks has more than it shows."""
        return self.scroll.verticalScrollBar().maximum() > 0

    def cards(self) -> list[TrackCard]:
        return [self._cards[tid] for tid in self._order]

    def card(self, track_id: str) -> Optional[TrackCard]:
        return self._cards.get(track_id) if track_id in self._order else None

    def track_ids(self) -> list[str]:
        return list(self._order)

    def _move(self, card: TrackCard, step: int) -> None:
        order = list(self._order)
        if card.track_id not in order:
            return
        i = order.index(card.track_id)
        j = i + step
        if not 0 <= j < len(order):
            return
        order[i], order[j] = order[j], order[i]
        self.order_changed.emit(order)

    # -- shared view -----------------------------------------------------------

    def set_x_range(self, lo: float, hi: float) -> None:
        self._x_range = (float(lo), float(hi))
        for card in self.cards():
            card.plot.setXRange(lo, hi, padding=0.0)

    def set_marker(self, x: Optional[float]) -> None:
        self._marker = x
        for card in self.cards():
            card.set_marker(x)

    def set_regions(self, spans) -> None:
        """Stretches to shade on every track (flagged segments)."""
        self._regions = list(spans or [])
        for card in self.cards():
            card.set_regions(self._regions)

    def set_window(self, span) -> None:
        """The stretch on screen above, outlined on every track."""
        self._window = span
        for card in self.cards():
            card.set_window(span)

    def _hover_at(self, x: Optional[float], source: Optional[TrackCard],
                  y: Optional[float] = None) -> None:
        when = self.describe_x(x) if x is not None else ""
        for card in self.cards():
            card.set_hover(x)
            if x is None:
                text = card.track.get("summary", "")
            else:
                value = card.read(x, y if card is source else None)
                text = f"{when}: {value}" if value else when
            if card.readout.text() != text:
                card.readout.setText(text)

    # -- size ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """``fit``: the tracks share the room (scrolling only below their
        least height); ``scroll``: each gets a readable height."""
        if mode not in ("fit", "scroll") or mode == self._mode:
            return
        self._mode = mode
        self._apply_mode()

    def mode(self) -> str:
        return self._mode

    def _apply_mode(self) -> None:
        cards = self.cards()
        for k, card in enumerate(cards):
            h = max(1.0, float(card.track.get("height", 1)))
            last = k == len(cards) - 1
            extra = AXIS_PX if last else 0
            if self._mode == "scroll":
                card.setFixedHeight(int(SCROLL_PX * h) + extra)
            else:
                card.setMinimumHeight(int(FIT_PX * h) + extra)
                card.setMaximumHeight(16_777_215)
            self._lay.setStretch(k, int(round(h * 10)) if self._mode == "fit" else 0)
            card.show_x_axis(last, self._x_label)
        self.request_align()

    def wanted_height(self) -> int:
        """The height that shows every track at its fitted size."""
        h = sum(max(1.0, float(c.track.get("height", 1))) for c in self.cards())
        return int(FIT_PX * h + AXIS_PX + 12 + 6 * max(len(self._order) - 1, 0))

    # -- look ------------------------------------------------------------------

    def apply_theme(self) -> None:
        for card in self._cards.values():
            if card.track:
                card.apply_theme()
                card.set_regions(self._regions)
                card.set_window(self._window)


__all__ = ["FIT_PX", "SCROLL_PX", "TrackCard", "TracksPanel"]
