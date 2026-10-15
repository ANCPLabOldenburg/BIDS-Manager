"""The traces canvas: a stretch of a multichannel signal, one trace per channel.

The drawing half of what was ``TimeSeriesView``, now driven by the scene
(``Scene.traces``) and a :class:`~bidsmgr.viz.data.signal.SignalSource`, so
MEG, EEG, iEEG and physio are one canvas and every control is a command.

What it keeps from the old view, because each was a measured lesson:

* every sample is read and drawn through MIN/MAX decimation per pixel
  column, so a one-sample spike survives and Fit all does not freeze;
* a gap is drawn as a break (``connect="finite"``), never as the zero the
  array holds;
* filtering pads the segment within a sample budget and refuses a cut-off
  the recording cannot resolve, saying so (``viz.compute.filters``);
* above two traces nothing wider than one pixel is drawn, whatever was asked:
  Qt strokes a wider pen through its general path code, 79.8 ms against
  11.3 ms for eight traces, and a drag pays that on every frame;
* a drag scrubs through the RECORDING (the next stretch is read), paced by
  the cost of the last redraw so a wall of MEG still follows the cursor;
* channel names are placed by mapping each band's edges through the view,
  so a name sits level with its trace whatever the axis and margins.

New: curve items are REUSED between redraws (the old view cleared and rebuilt
every one), the mouse goes through the configurable mouse map, and traces are
drawn in RUN time (``start_time`` added) so physio events line up.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsRectItem,
    QHBoxLayout,
    QScrollBar,
    QToolTip,
    QWidget,
)

from ....viz import inputmap
from ....viz.commands import signal as sigcmd
from ....viz.compute.decimate import peak_decimate
from ....viz import colorbar
from ....viz.compute.filters import FilterSpec, filter_recording, fits_in_memory, segment
from ....viz.theme import parse_colour
from .. import fonts
from ..bridge import connect_while_alive
from ..context import ViewerContext
from ...widgets.splitter import Splitter

log = logging.getLogger(__name__)

#: Trace width when only a few traces are on screen. A hairline reads as a
#: scratch on a modern display.
THICK_LINE_WIDTH = 2
#: Above this many traces nothing wider than one pixel is drawn (see the
#: module docstring for the measurement).
THICK_LINE_MAX_TRACES = 2
#: Events drawn at most; beyond it they are thinned evenly.
MAX_EVENTS_DRAWN = 500
#: Samples the centre of a trace is estimated from. Its range is exact (the
#: envelope keeps every extreme); its mean only places it in its band, and
#: two hundred thousand evenly spread samples place it identically.
_MEAN_SAMPLES = 200_000
#: Pixels reserved above the top trace for event labels.
_EVENT_LABEL_PX = 16.0


def window_envelopes(src, shown: list[int], t0: float, t1: float, spec: FilterSpec,
                     width_px: int, copy=None):
    """Worker side (or inline when cheap): the window read, filtered,
    decimated to about two points a pixel. ``(envelopes, segment)``."""
    seg = segment(src, shown, t0, t1, spec, copy=copy)
    if seg is None:
        return None, None
    out = []
    for trace in seg.data:
        xd, env = peak_decimate(seg.times, trace, width_px)
        step = max(1, trace.size // _MEAN_SAMPLES)
        out.append((np.asarray(xd, dtype=float), np.asarray(env, dtype=float),
                    _nan_mean(trace[::step]), _nan_ptp(env)))
    return out, seg


def _nan_mean(values) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else 0.0


def _nan_ptp(values) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max() - finite.min()) if finite.size else 0.0


def _qcolor(value: str) -> QColor:
    # QColor("rgba(...)") is black; the palette may use either spelling.
    return QColor(*parse_colour(value))


class _LabelStrip(QWidget):
    """The channel names, painted level with their traces.

    ONE widget whatever the channel count. It was a QLabel per channel, and
    showing a hundred cost 0.8 s on the GUI thread, each label polished
    against the app stylesheet as it appeared and laid out again on every
    redraw: a stall on every change of channel count, and 0.3 s of every
    MEG load. Where the bands are too thin for every name, every k-th is
    drawn rather than all of them on top of each other; the tooltip names
    whichever band the pointer is over.
    """

    WIDTH = 68
    _PIXEL_SIZE = 10

    #: A name was clicked (its band's index).
    clicked = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.names: list[str] = []
        #: Bands whose channel is bad: painted in the error colour.
        self.bad_rows: set = set()
        #: Bands the quality check flagged: ``{row: (colour, reason text)}``,
        #: a marker beside the name and the reasons in its tooltip.
        self.flags: dict[int, tuple[QColor, str]] = {}
        #: The band a QC channel map sent you to: its name always drawn,
        #: bold, on a tinted band (never skipped for room).
        self.focus_row: Optional[int] = None
        self._bad = QColor("#f85149")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._edges: list[int] = []
        self._bg = QColor("#000000")
        self._fg = QColor("#cccccc")
        self._font = fonts.font(self._PIXEL_SIZE)

    def set_colours(self, bg: QColor, fg: QColor, bad: Optional[QColor] = None) -> None:
        # Re-made with the colours: a font-size change re-applies the theme.
        self._font = fonts.font(self._PIXEL_SIZE)
        if bad is not None:
            self._bad = bad
        if bg != self._bg or fg != self._fg:
            self._bg, self._fg = bg, fg
            self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            i = self.band_at(event.position().y())
            if i is not None:
                self.clicked.emit(i)

    def set_bands(self, names: list[str], edges: list[int]) -> None:
        """``edges``: n + 1 y positions, name i between edges i and i + 1."""
        self.names = list(names)
        self._edges = list(edges) if len(edges) == len(names) + 1 else []
        self.update()

    def band_at(self, y: float) -> Optional[int]:
        for i in range(len(self._edges) - 1):
            if self._edges[i] <= y < self._edges[i + 1]:
                return i
        return None

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), self._bg)
        n = len(self.names)
        if not n or not self._edges:
            return
        p.setFont(self._font)
        p.setPen(self._fg)
        fm = QFontMetrics(self._font)
        line = fm.height()
        band = (self._edges[-1] - self._edges[0]) / n
        stride = max(1, math.ceil(line / max(band, 1e-6)))
        width = self.width() - 6
        align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for i in range(0, n, stride):
            centre = (self._edges[i] + self._edges[i + 1]) / 2.0
            height = max(line, self._edges[i + 1] - self._edges[i])
            rect = QRect(2, int(round(centre - height / 2.0)), width, int(height))
            p.setPen(self._bad if i in self.bad_rows else self._fg)
            name_width = width - (8 if i in self.flags else 0)
            p.drawText(rect, align, fm.elidedText(self.names[i], Qt.TextElideMode.ElideMiddle,
                                                  name_width))
            if i in self.flags:
                p.save()
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(self.flags[i][0])
                text_w = fm.horizontalAdvance(fm.elidedText(
                    self.names[i], Qt.TextElideMode.ElideMiddle, name_width))
                x = max(3.0, 2 + width - text_w - 8.0)
                p.drawEllipse(QPointF(x, centre), 2.6, 2.6)
                p.restore()
        i = self.focus_row
        if i is not None and 0 <= i < n:
            centre = (self._edges[i] + self._edges[i + 1]) / 2.0
            height = max(line, self._edges[i + 1] - self._edges[i])
            band_rect = QRect(0, int(round(centre - height / 2.0)), self.width(), int(height))
            # Covers whatever a neighbour drew there when names are thinned.
            p.fillRect(band_rect, self._bg)
            tint = QColor(self._fg)
            tint.setAlpha(40)
            p.fillRect(band_rect, tint)
            bold = QFont(self._font)
            bold.setBold(True)
            p.setFont(bold)
            p.setPen(self._bad if i in self.bad_rows else self._fg)
            rect = QRect(2, band_rect.top(), width, band_rect.height())
            p.drawText(rect, align, QFontMetrics(bold).elidedText(
                self.names[i], Qt.TextElideMode.ElideMiddle, width))

    def event(self, event) -> bool:  # noqa: D401
        if event.type() == QEvent.Type.ToolTip:
            i = self.band_at(event.pos().y())
            if i is None:
                QToolTip.hideText()
            else:
                state = "bad" if i in self.bad_rows else "good"
                text = (f"{self.names[i]} ({state}): click to mark it "
                        f"{'good' if state == 'bad' else 'bad'}")
                if i in self.flags:
                    text += f"\nQC: {self.flags[i][1]}"
                QToolTip.showText(event.globalPos(), text, self)
            return True
        return super().event(event)


class TracesCanvas(QWidget):
    """Label strip, plot, channel scroll bar."""

    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setObjectName("pane-dark")
        import pyqtgraph as pg

        self._pg = pg
        self._curves: list = []
        # Event markers are POOLED: reused and hidden, never removed (see
        # _pooled).
        self._event_lines: list = []
        self._event_spans: list = []
        self._event_texts: list = []
        self._shown: list[int] = []
        self._offsets: list[float] = []
        self._last_messages: tuple = ()
        # The decimated envelopes of the window on screen. A change of scale,
        # normalisation, colour or theme redraws from these without reading
        # or decimating a sample.
        self._env_key: Optional[tuple] = None
        self._env: list = []
        # Filtering: a filtered copy of the whole recording when it fits in
        # memory (made once, on a worker), else each window on a worker.
        self._copy = None
        self._copy_src = None
        self._copy_wanted = None
        self._window_wanted = None
        self._seg = None
        self._preview = False
        # Scale bars, pooled like every overlay item.
        self._bars = None
        self._bar_texts: list = []
        self._drag_t0: Optional[float] = None
        self._drag_x: float = 0.0
        self._drag_cost = 0.0
        self._drag_last = 0.0
        self._wheel_acc = 0.0
        # The time cursor: one line, made on first use, never removed.
        self._cursor_line = None
        # The stretch a QC channel map sent you to: a halo and a line, made
        # on first use, hidden and reused, never removed (CLAUDE.md guard 8d).
        self._focus_items: list = []
        # Bad segments (pooled regions and their labels), the segment being
        # drawn in annotation mode, and a guard while regions are placed.
        self._bad_regions: list = []
        self._bad_texts: list = []
        self._draft = None
        self._draft_t0: Optional[float] = None
        self._placing = False

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        split = Splitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        self._strip = _LabelStrip()

        self.plot = pg.PlotWidget()
        self.plot.setMinimumWidth(60)
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        pi = self.plot.getPlotItem()
        pi.getAxis("bottom").enableAutoSIPrefix(False)
        pi.getAxis("left").setWidth(0)
        pi.getAxis("left").setStyle(showValues=False)
        pi.setMenuEnabled(False)
        pi.hideButtons()
        # pyqtgraph's own panning is OFF: dragging moves the window through
        # the RECORDING (the next stretch is read), not the camera over the
        # stretch that was fetched.
        self.plot.setMouseEnabled(x=False, y=False)
        self._vb = pi.getViewBox()
        self._vb.mouseDragEvent = self._on_drag
        self._vb.mouseClickEvent = self._on_click
        self.plot.wheelEvent = self._on_wheel
        self._vb.sigResized.connect(self._position_labels)
        self._hover = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30,
                                     slot=self._on_hover)

        self._strip.clicked.connect(self._on_name_clicked)
        split.addWidget(self._strip)
        split.addWidget(self.plot)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        row.addWidget(split, 1)
        self.scroll = QScrollBar(Qt.Orientation.Vertical)
        self.scroll.setToolTip("Scroll channels")
        self.scroll.valueChanged.connect(self._on_scroll)
        row.addWidget(self.scroll)

        ctx.qstore.changed.connect(self._on_changed)
        ctx.jobs.done.connect(self._on_job_done)
        ctx.jobs.failed.connect(self._on_job_failed)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.redraw())
        connect_while_alive(ctx.settings_hub.changed, self, lambda w, _s: w.redraw())
        self._apply_theme()

    # ------------------------------------------------------------------
    @property
    def source(self):
        return sigcmd.source(self.ctx.store)

    def _on_changed(self, paths) -> None:
        if any(p == "scene" or p.startswith("sources") for p in paths):
            # A new or resampled recording: the cached window describes the
            # old one (and would keep its samples alive).
            self._env_key, self._env = None, []
            self._copy = self._copy_wanted = self._window_wanted = None
            self._seg = None
        if not self.isVisible():
            return
        if any(p.startswith("traces") or p == "scene" or p.startswith("sources")
               for p in paths):
            self.redraw()
        elif "cursor.time" in paths:
            self._draw_cursor()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.redraw()

    # ------------------------------------------------------------------
    # Theme and pens
    # ------------------------------------------------------------------

    def _dark_override(self) -> bool:
        return not self.ctx.theme.dark and self.ctx.settings.traces.dark_plot

    def _apply_theme(self) -> None:
        theme = self.ctx.theme
        if self._dark_override():
            bg, fg = "#11161d", "#cccccc"
        else:
            bg, fg = theme.plot_background, theme.plot_foreground
        self.plot.setBackground(bg)
        pi = self.plot.getPlotItem()
        for name in ("bottom", "left"):
            pi.getAxis(name).setPen(self._pg.mkPen(color=fg))
        fonts.style_axes(pi, fg)
        fonts.axis_title(pi, "bottom", "Time", fg, units="s")
        # Labels drawn earlier keep the font they were made with: re-made at
        # the app's size now (a font-size change re-applies the theme).
        small = fonts.font(fonts.SMALL_PX)
        for txt in (*self._event_texts, *self._bad_texts):
            txt.setFont(small)
        for txt in self._bar_texts:
            txt.setFont(fonts.font(fonts.LABEL_PX))
        self._label_metrics = None

    def max_pen_width(self, n_shown: Optional[int] = None) -> int:
        """The widest pen this view draws at: a CAP, applied to a chosen
        width too (a six-pixel preference must not turn a twenty-channel
        window into a slideshow)."""
        if n_shown is None:
            n_shown = max(1, len(self._shown))
        return THICK_LINE_WIDTH if n_shown <= THICK_LINE_MAX_TRACES else 1

    def pen_width(self, n_shown: Optional[int] = None) -> int:
        if self.max_pen_width(n_shown) <= 1:
            return 1
        chosen = self.ctx.settings.traces.line_width
        return chosen if chosen > 0 else THICK_LINE_WIDTH

    def colour_for(self, ch_type: str) -> str:
        ts = self.ctx.settings.traces
        return ts.line_color or self.ctx.theme.type_colour(ch_type, ts.type_colors)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def shown_channels(self) -> list[int]:
        """Indices of the channels drawn now."""
        src = self.source
        tr = self.ctx.scene.traces
        if src is None:
            return []
        pool = src.picks_for(tr.ch_type, tr.picks)
        start = min(tr.offset, max(0, len(pool) - tr.count))
        return pool[start:start + tr.count]

    def redraw(self) -> None:
        if not self.isVisible():
            return
        self._apply_theme()
        src = self.source
        tr = self.ctx.scene.traces
        self._clear_overlay()
        if src is None:
            self._env_key, self._env = None, []
            self._set_curves(0)
            self._set_labels([])
            return
        pool = src.picks_for(tr.ch_type, tr.picks)
        self._update_scrollbar(len(pool))
        shown = self.shown_channels()
        self._shown = shown
        n = len(shown)
        if n == 0:
            self._set_curves(0)
            self._set_labels([])
            return
        t0 = tr.t0
        t1 = min(tr.t0 + tr.width, src.duration)
        width_px = max(1, int(self._vb.width()) or self.plot.width())
        envs = self._envelopes(src, shown, t0, t1, FilterSpec(tr.hp, tr.lp, tr.notch), width_px)
        if envs is None:
            return
        refs = self._refs(src, shown, envs, tr)
        bads = sigcmd.bad_channels(self.ctx.store)
        # Butterfly: one band per channel TYPE, its channels overlaid.
        if tr.butterfly:
            groups = list(dict.fromkeys(src.ch_types[ch] for ch in shown))
            bands = len(groups)
            row_of = {t: bands - 1 - k for k, t in enumerate(groups)}
        else:
            groups, bands, row_of = [], n, {}
        pen_w = self.pen_width(n)
        self._set_curves(n)
        self._offsets = []
        dim = _qcolor(self.ctx.theme.dim)
        dim.setAlpha(150)
        for i, ((xd, env, mean, spread), ch) in enumerate(zip(envs, shown)):
            ch_type = src.ch_types[ch]
            if tr.normalize:
                ref = spread if spread > 0 else 1.0
            else:
                ref = refs.get(ch_type, 1.0) or 1.0
            offset = row_of[ch_type] if tr.butterfly else n - 1 - i
            centre = mean if tr.remove_dc else 0.0
            # Scaled AFTER decimation: a min/max envelope scales exactly, and
            # it is two points a pixel rather than every sample.
            rel = (env - centre) / ref * tr.scale
            if tr.clip:
                # An artefact stays in its neighbourhood (MNE clips at 1.5).
                rel = np.clip(rel, -1.5, 1.5)
            yd = rel + offset
            curve = self._curves[i]
            bad = src.ch_names[ch] in bads
            colour = dim if bad else self.colour_for(ch_type)
            curve.setPen(self._pg.mkPen(color=colour, width=pen_w))
            curve.setZValue(-1 if bad else 0)
            curve.setData(xd + src.start_time, yd, connect="finite")
            self._offsets.append(offset)
        pi = self.plot.getPlotItem()
        pi.setXRange(t0 + src.start_time, t1 + src.start_time, padding=0)
        # Room above the top trace for the event labels, in pixels: placed
        # in data units alone they sat above the visible range and showed
        # as a clipped sliver.
        headroom = 0.0
        if tr.events:
            px = float(self._vb.height())
            headroom = _EVENT_LABEL_PX * bands * 1.04 / max(px - _EVENT_LABEL_PX * 1.04, 1.0)
        top = bands - 0.5 + headroom
        pi.setYRange(-0.5, top, padding=0.02)
        if tr.butterfly:
            # Name i labels the i-th band from the TOP: the first group.
            self._set_labels([src.type_label(t) for t in groups], set())
        else:
            self._set_labels([src.ch_names[c] for c in shown],
                             {k for k, c in enumerate(shown) if src.ch_names[c] in bads})
        if not tr.normalize:
            self._draw_scale_bars(src, shown, refs, tr, t1 + src.start_time, row_of)
        if tr.events:
            self._draw_events(t0 + src.start_time, t1 + src.start_time, top)
        self._draw_bad_spans(t0 + src.start_time, t1 + src.start_time, top)
        focus_row = self._draw_focus(src, shown, row_of, tr)
        if self._strip.focus_row != focus_row:
            self._strip.focus_row = focus_row
            self._strip.update()
        self._draw_cursor()
        self.plot.setCursor(Qt.CursorShape.CrossCursor if tr.annotate
                            else Qt.CursorShape.ArrowCursor)

    def _refs(self, src, shown, envs, tr) -> dict:
        """Data units per band, per channel type: the recording's own
        (measured once), or this page's when asked."""
        if tr.normalize:
            return {}
        if tr.page_scale:
            spreads: dict[str, list[float]] = {}
            for (_x, _env, _mean, spread), ch in zip(envs, shown):
                spreads.setdefault(src.ch_types[ch], []).append(spread)
            return {kind: (float(np.median([v for v in values if v > 0]))
                           if any(v > 0 for v in values) else 1.0)
                    for kind, values in spreads.items()}
        return src.type_scales()

    def _draw_scale_bars(self, src, shown, refs, tr, x_right: float, row_of: dict) -> None:
        """A bar per channel type at the right edge, a round number of its
        unit tall ("200 fT", "50 µV"), level with that type's first trace."""
        pg = self._pg
        if self._bars is None:
            self._bars = pg.PlotCurveItem(connect="pairs", antialias=False)
            self._bars.setZValue(30)
            self.plot.getPlotItem().addItem(self._bars)
        xs, ys, labels = [], [], []
        seen = set()
        n = len(shown)
        for i, ch in enumerate(shown):
            ch_type = src.ch_types[ch]
            if ch_type in seen:
                continue
            seen.add(ch_type)
            unit, factor = src.unit_for(ch_type)
            per_band = (refs.get(ch_type, 1.0) or 1.0) / max(tr.scale, 1e-12) * factor
            # Round DOWN: a bar taller than its band reads as a trace.
            value = colorbar.nice_floor(per_band * 0.6)
            height = value / per_band
            row = row_of.get(ch_type, n - 1 - i)
            x = x_right - (x_right - self._vb.viewRange()[0][0]) * 0.012
            xs += [x, x]
            ys += [row - height / 2, row + height / 2]
            labels.append((x, row, f"{colorbar.fmt(value, value)} {unit}"))
        self._bars.setData(np.asarray(xs, float), np.asarray(ys, float))
        self._bars.setPen(pg.mkPen(color=self.ctx.theme.text, width=2))
        def bar_text():
            item = pg.TextItem("", anchor=(1.0, 0.5))
            item.setFont(fonts.font(fonts.LABEL_PX))
            return item

        texts = self._pooled(self._bar_texts, bar_text, len(labels))
        fill = _qcolor(self.ctx.theme.plot_background)
        fill.setAlpha(210)
        for txt, (x, row, text) in zip(texts, labels):
            txt.setText(text, color=self.ctx.theme.text)
            # On the plot's own colour: the traces run under the label.
            txt.fill = pg.mkBrush(fill)
            txt.setZValue(31)
            txt.setPos(x, row)

    def _envelopes(self, src, shown: list[int], t0: float, t1: float,
                   spec: FilterSpec, width_px: int) -> Optional[list]:
        """Per shown channel ``(x, envelope, mean, spread)`` of the window.

        Unfiltered, read inline (a slice of a preloaded array). Filtered,
        NEVER on the GUI thread (it cost a second a page at 306 channels):
        the whole recording is filtered once on a worker when it fits in
        memory, and then a window is a slice; otherwise each window is
        filtered on a worker. Until the filtered result arrives the window
        is drawn unfiltered and the status bar says so.
        """
        copy = self._copy if (spec.active and self._copy is not None
                              and self._copy.spec == spec and self._copy_src is src) else None
        key = (id(src), tuple(shown), t0, t1, spec, width_px, id(copy))
        if key == self._env_key:
            return self._env
        if spec.active and copy is None:
            self._request_filter(src, shown, t0, t1, spec, width_px, key)
            envs, seg = window_envelopes(src, shown, t0, t1, FilterSpec(), width_px)
            if envs is None:
                return None
            self._preview = True
            self._seg = seg
            return envs
        envs, seg = window_envelopes(src, shown, t0, t1, spec, width_px, copy=copy)
        if envs is None:
            return None
        self._preview = False
        messages = tuple(seg.messages)
        if messages and messages != self._last_messages:
            # Said once per change, not on every redraw.
            self.ctx.status.emit(" ".join(messages))
        self._last_messages = messages
        self._seg = seg
        self._env_key, self._env = key, envs
        return envs

    def _request_filter(self, src, shown, t0, t1, spec, width_px, key) -> None:
        if fits_in_memory(src, spec):
            wanted = (id(src), spec)
            if self._copy_wanted != wanted:
                self._copy_wanted = wanted
                self._copy_src = src
                self.ctx.status.emit(f"Filtering the recording ({spec.describe()})...")
                self.ctx.jobs.start("traces-filter", hash(wanted) & 0x7FFFFFFF,
                                    filter_recording, src, spec)
            return
        if self._window_wanted != key:
            self._window_wanted = key
            self.ctx.jobs.start("traces-window", hash(key) & 0x7FFFFFFF, window_envelopes,
                                src, list(shown), t0, t1, spec, width_px)

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if tag == "traces-filter" and self._copy_wanted is not None \
                and generation == (hash(self._copy_wanted) & 0x7FFFFFFF):
            self._copy = result
            self._env_key = None
            if result.messages:
                self.ctx.status.emit(" ".join(result.messages))
            self.redraw()
        elif tag == "traces-window" and self._window_wanted is not None \
                and generation == (hash(self._window_wanted) & 0x7FFFFFFF):
            envs, seg = result
            if envs is not None:
                self._env_key, self._env, self._seg = self._window_wanted, envs, seg
                self._preview = False
                if seg.messages:
                    self.ctx.status.emit(" ".join(seg.messages))
            self._window_wanted = None
            self.redraw()

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if tag in ("traces-filter", "traces-window"):
            self._copy_wanted = self._window_wanted = None
            self.ctx.status.emit(f"The filter could not be applied: {message}")

    def is_preview(self) -> bool:
        """Whether the window is drawn unfiltered while the filter runs."""
        return self._preview

    def _on_name_clicked(self, row: int) -> None:
        if self.ctx.scene.traces.butterfly or not 0 <= row < len(self._shown):
            return
        src = self.source
        if src is not None:
            self.ctx.run("channels.toggle_bad", name=src.ch_names[self._shown[row]])

    def _set_curves(self, n: int) -> None:
        pi = self.plot.getPlotItem()
        while len(self._curves) < n:
            # Antialiasing per item, never process-wide.
            curve = self._pg.PlotCurveItem(antialias=False)
            pi.addItem(curve)
            self._curves.append(curve)
        for i, curve in enumerate(self._curves):
            if i >= n:
                curve.setData([], [])
            curve.setVisible(i < n)

    def _pooled(self, pool: list, make, k: int) -> list:
        """The first ``k`` items of ``pool`` shown, the rest hidden.

        Overlay items are reused, never removed. pyqtgraph takes an item out
        of the scene BEFORE detaching it from its view, and in between Qt
        can ask an event line for its bounds, with no view to measure them
        against; the item then freed under the scene segfaulted about one
        test run in seven. Reusing them is also cheaper while scrolling."""
        while len(pool) < k:
            item = make()
            self.plot.addItem(item, ignoreBounds=True)
            pool.append(item)
        for i, item in enumerate(pool):
            item.setVisible(i < k)
        return pool[:k]

    def _clear_overlay(self) -> None:
        if self._bars is not None:
            self._bars.setData([], [])
        for txt in self._bar_texts:
            txt.setVisible(False)
        for pool in (self._event_lines, self._event_spans, self._event_texts,
                     self._bad_regions, self._bad_texts):
            for item in pool:
                item.setVisible(False)

    def _new_text(self):
        # Hung from the top edge, into the room the redraw keeps for it.
        txt = self._pg.TextItem("", anchor=(0.5, 0.0))
        txt.setFont(fonts.font(fonts.SMALL_PX))
        return txt

    def _draw_events(self, x0: float, x1: float, top: float) -> None:
        src = self.source
        tr = self.ctx.scene.traces
        # Bad segments are drawn by _draw_bad_spans, from the review.
        events = [e for e in src.events(tr.event_source)
                  if x0 <= e.onset <= x1 and getattr(e, "kind", "") != "bad"]
        if len(events) > MAX_EVENTS_DRAWN:
            events = events[:: len(events) // MAX_EVENTS_DRAWN]
        labels = sorted({e.label for e in events})
        ts = self.ctx.settings.traces
        theme = self.ctx.theme
        pg = self._pg
        spans = [e for e in events if e.duration > 0]
        texts = self._labels_that_fit([e for e in events if e.label], x0, x1)
        lines = self._pooled(self._event_lines,
                             lambda: pg.InfiniteLine(angle=90, movable=False), len(events))
        regions = self._pooled(
            self._event_spans,
            lambda: pg.LinearRegionItem(values=(0, 1), movable=False), len(spans))
        labels_drawn = self._pooled(self._event_texts, self._new_text, len(texts))

        def colour_of(e) -> str:
            if getattr(e, "kind", "") == "bad":
                return theme.token("error", "#f85149")
            return ts.event_color or theme.series(labels.index(e.label))

        for line, e in zip(lines, events):
            line.setPen(pg.mkPen(color=colour_of(e), width=ts.event_width))
            line.setPos(e.onset)
        # An event that lasts: a thin band just under the row of labels.
        # Filled the full height, a task of 4 s trials painted every page
        # over, and bad segments (red, full height) read as tinted.
        h = max(float(self._vb.height()), 1.0)
        band = (max(0.0, 1.0 - (_EVENT_LABEL_PX + 5.0) / h), max(0.0, 1.0 - _EVENT_LABEL_PX / h))
        for region, e in zip(regions, spans):
            colour = colour_of(e)
            region.setRegion((e.onset, e.onset + e.duration))
            region.setSpan(*band)
            region.setBrush(pg.mkBrush(_qcolor(colour).name() + "b0"))
            for edge in region.lines:
                edge.setPen(pg.mkPen(color=colour, width=1))
                edge.setSpan(*band)
            region.setZValue(-10)
        for txt, e in zip(labels_drawn, texts):
            txt.setText(str(e.label), color=colour_of(e))
            txt.setPos(e.onset, top)

    # -- bad segments -------------------------------------------------------

    def _new_bad_region(self):
        pg = self._pg
        region = pg.LinearRegionItem(values=(0, 1), movable=False)
        region.setZValue(-8)
        region.sigRegionChangeFinished.connect(lambda r=region: self._on_region_moved(r))
        region.mouseClickEvent = lambda ev, r=region: self._on_region_clicked(r, ev)
        region._span_index = -1
        return region

    def _draw_bad_spans(self, x0: float, x1: float, top: float) -> None:
        """The bad segments in view: red, labelled, always shown (they are
        what an analysis leaves out). In annotation mode they move and
        resize under the mouse; the selected one is drawn stronger."""
        tr = self.ctx.scene.traces
        spans = sigcmd.bad_spans(self.ctx.store)
        shown = [(i, sp) for i, sp in enumerate(spans)
                 if sp.onset + sp.duration >= x0 and sp.onset <= x1]
        regions = self._pooled(self._bad_regions, self._new_bad_region, len(shown))
        texts = self._pooled(self._bad_texts, self._new_text, len(shown))
        red = self.ctx.theme.token("error", "#f85149")
        pg = self._pg
        self._placing = True
        try:
            for region, text, (i, sp) in zip(regions, texts, shown):
                selected = tr.annotate and tr.selected_span == i
                region._span_index = i
                region.setMovable(bool(tr.annotate))
                region.setRegion((sp.onset, sp.onset + sp.duration))
                region.setBrush(pg.mkBrush(_qcolor(red).name() + ("58" if selected else "30")))
                region.setHoverBrush(pg.mkBrush(_qcolor(red).name() + "48"))
                for edge in region.lines:
                    edge.setPen(pg.mkPen(color=red, width=3 if selected else 1))
                    edge.setHoverPen(pg.mkPen(color=red, width=3))
                text.setText(sp.label, color=red)
                text.setPos(max(sp.onset, x0) + 0.002 * (x1 - x0), top)
                text.setAnchor((0.0, 0.0))
        finally:
            self._placing = False

    def _on_region_moved(self, region) -> None:
        """A segment dragged or resized in annotation mode: one undoable
        change of the review."""
        if self._placing or region._span_index < 0:
            return
        lo, hi = region.getRegion()
        try:
            self.ctx.run("annotate.set", index=int(region._span_index), onset=float(lo),
                         duration=float(hi - lo))
        except ValueError as exc:
            self.ctx.status.emit(str(exc))
            self.redraw()

    def _on_region_clicked(self, region, event) -> None:
        tr = self.ctx.scene.traces
        if not tr.annotate or region._span_index < 0:
            event.ignore()
            return
        event.accept()
        self.ctx.run("annotate.select", index=int(region._span_index))
        if event.button() == Qt.MouseButton.RightButton:
            self._span_menu(int(region._span_index))

    def _span_menu(self, index: int) -> None:
        """Right-click on a segment: relabel it, or delete it."""
        from ..menus import popup_menu, submenu

        menu = popup_menu(self)
        relabel = submenu(menu, "Label")
        current = sigcmd.bad_spans(self.ctx.store)[index].label
        for label in sigcmd.BAD_LABELS:
            act = relabel.addAction(label)
            act.setCheckable(True)
            act.setChecked(label == current)
            act.triggered.connect(lambda _c=False, lab=label: self.ctx.run(
                "annotate.set", index=index, label=lab))
        delete = menu.addAction("Delete this segment")
        delete.triggered.connect(lambda: self.ctx.run("annotate.remove", index=index))
        menu.exec(QCursor.pos())

    def _draw_draft(self, t0: Optional[float], t1: Optional[float]) -> None:
        """The segment being drawn, while the drag lasts."""
        pg = self._pg
        if self._draft is None:
            if t0 is None:
                return
            self._draft = pg.LinearRegionItem(values=(0, 1), movable=False)
            self._draft.setZValue(-7)
            self.plot.addItem(self._draft, ignoreBounds=True)
        self._draft.setVisible(t0 is not None)
        if t0 is not None:
            red = self.ctx.theme.token("error", "#f85149")
            self._draft.setBrush(pg.mkBrush(_qcolor(red).name() + "40"))
            for edge in self._draft.lines:
                edge.setPen(pg.mkPen(color=red, width=2, style=Qt.PenStyle.DashLine))
            self._draft.setRegion((min(t0, t1), max(t0, t1)))

    def bad_region_items(self) -> list:
        """The bad-segment regions on screen (tests)."""
        return [r for r in self._bad_regions if r.isVisible()]

    def _labels_that_fit(self, events: list, x0: float, x1: float) -> list:
        """The events whose label has room: a label that would overlap the
        one before it is left out (its line is still drawn), or two triggers
        a few milliseconds apart printed one name over the other."""
        if not events:
            return []
        fm = getattr(self, "_label_metrics", None)
        if fm is None:
            fm = self._label_metrics = QFontMetrics(fonts.font(fonts.SMALL_PX))
        px_per_s = max(1.0, float(self._vb.width())) / max(x1 - x0, 1e-9)
        kept, right = [], -math.inf
        for e in sorted(events, key=lambda ev: ev.onset):
            x = (e.onset - x0) * px_per_s
            half = fm.horizontalAdvance(str(e.label)) / 2.0 + 3.0
            if x - half >= right:
                kept.append(e)
                right = x + half
        return kept

    def _draw_cursor(self) -> None:
        """The time cursor, a line in the text colour labelled with its
        time; hidden when there is none."""
        t = self.ctx.scene.cursor.time
        line = self._cursor_line
        if line is None:
            if t is None:
                return
            pg = self._pg
            line = pg.InfiniteLine(angle=90, movable=False, label="{value:.3f} s",
                                   labelOpts={"position": 0.04, "anchors": [(0, 1), (0, 1)]})
            line.setZValue(20)
            self.plot.addItem(line, ignoreBounds=True)
            self._cursor_line = line
        line.setVisible(t is not None and self.source is not None)
        if t is None:
            return
        # The text colour, not the accent: magnetometers are drawn in the
        # accent, and the cursor vanished into them.
        ink = _qcolor(self.ctx.theme.text)
        line.setPen(self._pg.mkPen(color=ink, width=1))
        line.label.setColor(ink)
        line.setPos(float(t))

    def _draw_focus(self, src, shown: list[int], row_of: dict, tr) -> Optional[int]:
        """The stretch a QC channel map sent you to, outlined on its channel
        (``TracesState.focus``): a line in the warning colour, which no
        channel type is drawn in and the QC flags already use, over a halo
        in the plot's background, so it stands out on any trace. Returns the
        channel's band, or None when it is not on screen."""
        focus = tr.focus
        band = row = None
        if focus is not None and focus.channel in src.ch_names:
            ch = src.ch_names.index(focus.channel)
            if ch in shown:
                i = shown.index(ch)
                if tr.butterfly:
                    band = row_of[src.ch_types[ch]]
                else:
                    band, row = len(shown) - 1 - i, i
        if band is None:
            for item in self._focus_items:
                if item.isVisible():
                    item.setVisible(False)
            return None
        if not self._focus_items:
            for z in (14, 15):
                item = QGraphicsRectItem()
                item.setZValue(z)
                self.plot.addItem(item, ignoreBounds=True)
                self._focus_items.append(item)
        theme = self.ctx.theme
        width = focus.duration if focus.duration > 0 else tr.width / 200.0
        rect = QRectF(float(focus.onset), band - 0.5, float(width), 1.0)
        for item, colour, px in ((self._focus_items[0], theme.plot_background, 5.0),
                                 (self._focus_items[1], theme.token("warning", "#d29922"), 2.0)):
            pen = QPen(_qcolor(colour), px)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            item.setRect(rect)
            item.setVisible(True)
        return row

    def focus_rect(self) -> Optional[QRectF]:
        """The outlined stretch in data units, when one is shown."""
        items = self._focus_items
        return items[1].rect() if items and items[1].isVisible() else None

    def cursor_visible(self) -> bool:
        return self._cursor_line is not None and self._cursor_line.isVisible()

    # -- labels ------------------------------------------------------------

    #: The quality check's verdict per channel name: ``{name: (token, text)}``.
    quality_flags: dict = {}

    def set_quality_flags(self, flags: dict) -> None:
        """Mark channels the quality check flagged (``{name: (theme token,
        reason text)}``), beside their names; empty clears them."""
        self.quality_flags = dict(flags)
        self.redraw()

    def _set_labels(self, names: list[str], bad_rows: Optional[set] = None) -> None:
        theme = self.ctx.theme
        self._strip.set_colours(_qcolor(theme.plot_background), _qcolor(theme.text),
                                _qcolor(theme.token("error", "#f85149")))
        self._strip.names = list(names)
        self._strip.bad_rows = set(bad_rows or ())
        flags = {}
        if self.quality_flags and not self.ctx.scene.traces.butterfly:
            for row, name in enumerate(names):
                hit = self.quality_flags.get(name)
                if hit is not None:
                    flags[row] = (_qcolor(theme.token(hit[0], "#d29922")), hit[1])
        self._strip.flags = flags
        self._position_labels()

    def _label_y(self, data_y: float) -> int:
        scene = self._vb.mapViewToScene(QPointF(0.0, float(data_y)))
        on_screen = self.plot.viewport().mapToGlobal(self.plot.mapFromScene(scene))
        return self._strip.mapFromGlobal(on_screen).y()

    def _position_labels(self, *_args) -> None:
        """Each name level with its trace: the band edges mapped through the
        view, so the axis and margins cannot put them out of step."""
        names = self._strip.names
        n = len(names)
        if not n:
            self._strip.set_bands([], [])
            return
        try:
            edges = [self._label_y(n - 0.5 - i) for i in range(n + 1)]
        except Exception:  # noqa: BLE001 - not laid out yet
            edges = []
        if len(edges) != n + 1 or edges[-1] <= edges[0]:
            step = self._strip.height() / n
            edges = [int(round(i * step)) for i in range(n + 1)]
        self._strip.set_bands(names, edges)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _update_scrollbar(self, total: int) -> None:
        tr = self.ctx.scene.traces
        visible = min(tr.count, total)
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, max(0, total - visible))
        self.scroll.setPageStep(max(1, visible))
        self.scroll.setValue(min(tr.offset, max(0, total - visible)))
        self.scroll.blockSignals(False)
        self.scroll.setVisible(total > visible)

    def _on_scroll(self, value: int) -> None:
        tr = self.ctx.scene.traces
        self.ctx.run("traces.scroll", n=int(value) - tr.offset)

    def _focus_viewer(self) -> None:
        w = self.parentWidget()
        while w is not None and not getattr(w, "_is_viz_viewer", False):
            w = w.parentWidget()
        if w is not None:
            w.setFocus(Qt.FocusReason.MouseFocusReason)

    def _mods(self, event) -> set[str]:
        m = event.modifiers()
        mods = set()
        if m & Qt.KeyboardModifier.ShiftModifier:
            mods.add("shift")
        if m & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            mods.add("ctrl")
        if m & Qt.KeyboardModifier.AltModifier:
            mods.add("alt")
        return mods

    def _on_drag(self, event, axis=None) -> None:
        """Drag scrubs the window through the recording, paced by cost.

        The next redraw waits at least twice as long as the last one took
        (the paint lands later and costs about as much again), never more
        often than 60 Hz; the release always redraws, so the view lands
        where the cursor left it."""
        src = self.source
        if (src is not None and self.ctx.scene.traces.annotate
                and event.button() == Qt.MouseButton.LeftButton):
            self._annotate_drag(event)
            return
        button = {Qt.MouseButton.LeftButton: "left", Qt.MouseButton.RightButton: "right",
                  Qt.MouseButton.MiddleButton: "middle"}.get(event.button(), "left")
        tool = inputmap.lookup("traces", button, self._mods(event), self.ctx.settings.mousemap)
        if src is None or src.duration <= 0 or tool != "scrub":
            event.ignore()
            return
        event.accept()
        tr = self.ctx.scene.traces
        per_px = tr.width / max(1, self._vb.width())
        if event.isStart():
            self._focus_viewer()
            self._drag_t0 = tr.t0
            self._drag_x = event.buttonDownPos().x()
            return
        if self._drag_t0 is None:
            return
        # Dragging LEFT moves forward in time, as paper under a finger.
        target = self._drag_t0 - (event.pos().x() - self._drag_x) * per_px
        finish = event.isFinish()
        now = time.perf_counter()
        if not finish and now - self._drag_last < max(2.0 * self._drag_cost, 1.0 / 60.0):
            return
        self.ctx.run("time.set", t0=target)
        self.ctx.qstore.flush()
        self._drag_cost = time.perf_counter() - now
        self._drag_last = time.perf_counter()
        if finish:
            self._drag_t0 = None

    def _annotate_drag(self, event) -> None:
        """Annotation mode: a drag across the traces marks a bad segment."""
        event.accept()
        t = float(self._vb.mapSceneToView(event.scenePos()).x())
        if event.isStart():
            self._focus_viewer()
            self._draft_t0 = float(self._vb.mapSceneToView(event.buttonDownScenePos()).x())
        if self._draft_t0 is None:
            return
        if event.isFinish():
            t0, self._draft_t0 = self._draft_t0, None
            self._draw_draft(None, None)
            try:
                self.ctx.run("annotate.add", onset=min(t0, t), duration=abs(t - t0))
            except ValueError as exc:
                self.ctx.status.emit(str(exc))
            return
        self._draw_draft(self._draft_t0, t)

    def _on_click(self, event) -> None:
        """A click without a drag places the time cursor (the status line
        reads the channel under it); a click on the cursor removes it."""
        src = self.source
        if src is None or event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        event.accept()
        self._focus_viewer()
        pt = self._vb.mapSceneToView(event.scenePos())
        t = float(pt.x())
        tr = self.ctx.scene.traces
        if tr.focus is not None:
            # You have looked: the outline goes, the click does what it does.
            self.ctx.run("traces.focus")
        if tr.annotate and tr.selected_span is not None:
            self.ctx.run("annotate.select", index=None)
            return
        per_px = tr.width / max(1.0, float(self._vb.width()))
        current = self.ctx.scene.cursor.time
        if current is not None and abs(current - t) <= 4.0 * per_px:
            self.ctx.run("cursor.time", t=None)
            self.ctx.status.emit("Time cursor removed")
            return
        self.ctx.run("cursor.time", t=t)
        row = self._row_at(pt.y())
        text = f"Cursor at {t:.3f} s"
        if row is not None and not tr.butterfly:
            ch = self._shown[row]
            text += f": {src.ch_names[ch]}{self._value_at(row, t - src.start_time)}"
        self.ctx.status.emit(text + " (click it again to remove it)")

    def _row_at(self, y: float) -> Optional[int]:
        if not self._offsets:
            return None
        best = min(range(len(self._offsets)), key=lambda i: abs(y - self._offsets[i]))
        return best if abs(y - self._offsets[best]) <= 0.6 else None

    def _value_at(self, row: int, t: float) -> str:
        """The value DRAWN at recording time ``t`` (filtered when a filter
        is on), in its unit; a gap is said to be one, never read as the zero
        the array holds there."""
        src = self.source
        seg = self._seg
        if src is None or seg is None or not seg.times.size or row >= seg.data.shape[0]:
            return ""
        k = int(round((t - float(seg.times[0])) * src.sfreq))
        if not 0 <= k < seg.data.shape[1]:
            return ""
        v = float(seg.data[row, k])
        unit, factor = src.unit_for(src.ch_types[self._shown[row]])
        return " = no sample (a gap)" if not np.isfinite(v) else f" = {v * factor:.4g} {unit}"

    def _on_wheel(self, event) -> None:
        self._focus_viewer()
        ad, pd = event.angleDelta(), event.pixelDelta()
        use_pixel = not pd.isNull()
        dx = pd.x() if use_pixel else ad.x()
        dy = pd.y() if use_pixel else ad.y()
        horizontal = abs(dx) > abs(dy)
        gesture = "hwheel" if horizontal else "wheel"
        tool = inputmap.lookup("traces", gesture, self._mods(event), self.ctx.settings.mousemap)
        delta = dx if horizontal else dy
        thresh = 40.0 if use_pixel else 120.0
        self._wheel_acc += delta
        steps = int(self._wheel_acc / thresh)
        self._wheel_acc -= steps * thresh
        if steps:
            if tool == "channels":
                self.ctx.run("traces.scroll", n=-steps)
            elif tool == "time":
                self.ctx.run("time.page", n=-steps * 0.25)
            elif tool == "zoom_time":
                self.ctx.run("time.zoom", factor=float(1.25 ** -steps))
            elif tool == "scale":
                self.ctx.run("traces.scale", factor=float(1.2 ** steps))
        event.accept()

    def _on_hover(self, evt) -> None:
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos) or not self._shown:
            return
        pt = self._vb.mapSceneToView(pos)
        src = self.source
        if src is None:
            return
        best = self._row_at(pt.y())
        if best is None:
            QToolTip.hideText()
            return
        # With a time cursor placed, the distance to it: an interval read
        # off the screen, as a latency or an inter-beat time.
        cursor = self.ctx.scene.cursor.time
        dt = f"  (Δt {pt.x() - cursor:+.3f} s)" if cursor is not None else ""
        if self.ctx.scene.traces.butterfly:
            QToolTip.showText(QCursor.pos(), f"{pt.x():.3f} s{dt}", self.plot)
            return
        ch = self._shown[best]
        value = self._value_at(best, pt.x() - src.start_time)
        QToolTip.showText(QCursor.pos(),
                          f"{src.ch_names[ch]} [{src.type_label(src.ch_types[ch])}]"
                          f"  {pt.x():.3f} s{value}{dt}", self.plot)

    # ------------------------------------------------------------------
    # Queries (tests)
    # ------------------------------------------------------------------

    def drawn(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The data of every visible curve (decimated as drawn)."""
        out = []
        for curve in self._curves[: len(self._shown)]:
            x, y = curve.getData()
            out.append((np.asarray(x), np.asarray(y)))
        return out

    def label_texts(self) -> list[str]:
        return list(self._strip.names)

    def overlay_items(self) -> list:
        return [item for pool in (self._event_spans, self._event_lines, self._event_texts)
                for item in pool if item.isVisible()]


__all__ = ["THICK_LINE_MAX_TRACES", "THICK_LINE_WIDTH", "TracesCanvas"]
