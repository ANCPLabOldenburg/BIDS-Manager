"""The power spectral density window, for every signal.

Two tabs, as before:

* **Per channel**: every channel of the chosen type overlaid (thin) with the
  type mean (bold), and a Highlight picker to isolate one channel;
* **Average (per type)**: each type's mean with a +/- 1 standard-deviation
  band and a legend.

Interactive (drag to pan, wheel or right-drag to zoom) with a crosshair and
a frequency / power readout, dB or linear. Colours come from the viewer
theme and the user's channel-type colours, so a recording's traces and its
spectrum agree; the window follows a theme swap like every other canvas.

It draws the result of :func:`bidsmgr.viz.compute.spectral.psd` and says
what it is: the raw signal, or the signal after the named filter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget,
)

from ....viz.compute.spectral import to_db
from .. import fonts
from ..bridge import SettingsHub, ThemeHub, connect_while_alive


class PsdWindow(QDialog):
    """Show one :func:`~bidsmgr.viz.compute.spectral.psd` result."""

    def __init__(self, result: dict, *, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(result.get("title") or "Power spectral density")
        self.setObjectName("pane-dark")
        self.resize(900, 560)
        import pyqtgraph as pg

        self._pg = pg
        self._freqs = np.asarray(result["freqs"])
        self._data = np.atleast_2d(np.asarray(result["data"]))
        self._names = list(result["ch_names"])
        self._types = list(result["ch_types"])
        n = min(self._data.shape[0], len(self._names), len(self._types))
        self._by_type: dict[str, list[int]] = {}
        for i in range(n):
            self._by_type.setdefault(self._types[i], []).append(i)
        self._db = True
        self._drawn: dict[int, list] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        top = QHBoxLayout()
        self.db_box = QCheckBox("dB (10 log10)")
        self.db_box.setChecked(True)
        self.db_box.toggled.connect(self._on_db)
        top.addWidget(self.db_box)
        what = (f"Filtered: {result['filter']}" if result.get("filtered")
                else "The raw signal, unfiltered")
        self.what = QLabel(what)
        self.what.setObjectName("sidecar-footer-summary")
        self.what.setToolTip("Which signal the spectrum was computed from.")
        top.addWidget(self.what)
        top.addStretch(1)
        hint = QLabel("Drag to pan, wheel or right-drag to zoom")
        hint.setObjectName("sidecar-footer-summary")
        top.addWidget(hint)
        outer.addLayout(top)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self._build_channel_tab()
        self._build_average_tab()
        connect_while_alive(ThemeHub.instance().changed, self, lambda w, _t: w.redraw())
        connect_while_alive(SettingsHub.instance().changed, self, lambda w, _s: w.redraw())
        self.redraw()

    # ------------------------------------------------------------------
    def _colour(self, ch_type: str) -> QColor:
        theme = ThemeHub.instance().theme
        return QColor(theme.type_colour(ch_type, SettingsHub.instance().settings.traces.type_colors))

    def _plot(self):
        pg = self._pg
        plot = pg.PlotWidget()
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
        return plot

    def _build_channel_tab(self) -> None:
        page = QWidget()
        page.setObjectName("pane-dark")
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.setObjectName("ent-input")
        for t in sorted(self._by_type):
            self.type_combo.addItem(f"{t}  ({len(self._by_type[t])})", t)
        self.type_combo.currentIndexChanged.connect(self._on_type)
        row.addWidget(self.type_combo)
        row.addWidget(QLabel("Highlight:"))
        self.highlight = QComboBox()
        self.highlight.setObjectName("ent-input")
        self.highlight.currentIndexChanged.connect(lambda _i: self._redraw_channels())
        row.addWidget(self.highlight, 1)
        v.addLayout(row)
        self.channel_plot = self._plot()
        v.addWidget(self.channel_plot, 1)
        self.channel_readout = QLabel("")
        self.channel_readout.setObjectName("sidecar-footer-summary")
        v.addWidget(self.channel_readout)
        self._channel_cross = self._crosshair(self.channel_plot, self.channel_readout)
        self.tabs.addTab(page, "Per channel")
        self._fill_highlight()

    def _build_average_tab(self) -> None:
        page = QWidget()
        page.setObjectName("pane-dark")
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        self.average_plot = self._plot()
        self.average_plot.addLegend(offset=(-10, 10))
        v.addWidget(self.average_plot, 1)
        self.average_readout = QLabel("")
        self.average_readout.setObjectName("sidecar-footer-summary")
        v.addWidget(self.average_readout)
        self._average_cross = self._crosshair(self.average_plot, self.average_readout)
        self.tabs.addTab(page, "Average (per type)")

    def _crosshair(self, plot, readout: QLabel) -> dict:
        pg = self._pg
        vline = pg.InfiniteLine(angle=90, movable=False)
        hline = pg.InfiniteLine(angle=0, movable=False)
        for line in (vline, hline):
            line.setZValue(10)
            plot.addItem(line, ignoreBounds=True)

        def moved(evt):
            pos = evt[0]
            if not plot.sceneBoundingRect().contains(pos):
                return
            pt = plot.getPlotItem().vb.mapSceneToView(pos)
            vline.setPos(pt.x())
            hline.setPos(pt.y())
            readout.setText(f"{pt.x():.2f} Hz   {pt.y():.3g} {self._ylabel()}")

        proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=30, slot=moved)
        return {"v": vline, "h": hline, "proxy": proxy}

    def _forget(self, plot) -> None:
        """Remove the curves this window drew, and nothing else.

        Not ``plot.clear()``: that takes the crosshair lines out too, and
        pyqtgraph removes an item from the scene before detaching it from its
        view, a window in which Qt can ask a line for bounds it cannot have.
        """
        for item in self._drawn.pop(id(plot), []):
            plot.removeItem(item)

    def _plot_curve(self, plot, values, **kwargs):
        item = plot.plot(self._freqs, values, **kwargs)
        self._drawn.setdefault(id(plot), []).append(item)
        return item

    def _ylabel(self) -> str:
        return "dB" if self._db else ""

    def _scale(self, arr):
        return to_db(arr) if self._db else arr

    def _fill_highlight(self) -> None:
        t = self.type_combo.currentData()
        self.highlight.blockSignals(True)
        self.highlight.clear()
        self.highlight.addItem("(none)", -1)
        for idx in self._by_type.get(t, []):
            self.highlight.addItem(self._names[idx], idx)
        self.highlight.blockSignals(False)

    def _on_db(self, on: bool) -> None:
        self._db = bool(on)
        self.redraw()

    def _on_type(self) -> None:
        self._fill_highlight()
        self._redraw_channels()

    # ------------------------------------------------------------------
    def redraw(self) -> None:
        theme = ThemeHub.instance().theme
        for plot in (self.channel_plot, self.average_plot):
            plot.setBackground(theme.plot_background)
            for name in ("left", "bottom"):
                plot.getPlotItem().getAxis(name).setPen(
                    self._pg.mkPen(color=theme.plot_foreground))
            fonts.style_axes(plot.getPlotItem(), theme.plot_foreground)
            fonts.axis_title(plot.getPlotItem(), "bottom", "Frequency",
                             theme.plot_foreground, units="Hz")
        pen = self._pg.mkPen(color=theme.dim, width=1, style=Qt.PenStyle.DashLine)
        for cross in (self._channel_cross, self._average_cross):
            cross["v"].setPen(pen)
            cross["h"].setPen(pen)
        self._redraw_channels()
        self._redraw_average()

    def _redraw_channels(self) -> None:
        pg = self._pg
        plot = self.channel_plot
        self._forget(plot)
        t = self.type_combo.currentData()
        idxs = self._by_type.get(t, [])
        if not idxs:
            return
        base = self._colour(t)
        thin = QColor(base)
        thin.setAlpha(60)
        for i in idxs:
            self._plot_curve(plot, self._scale(self._data[i]), pen=pg.mkPen(color=thin, width=1))
        self._plot_curve(plot, self._scale(np.mean(self._data[idxs], axis=0)),
                         pen=pg.mkPen(color=base, width=2))
        hi = self.highlight.currentData()
        if hi is not None and hi >= 0:
            self._plot_curve(plot, self._scale(self._data[hi]),
                             pen=pg.mkPen(color=ThemeHub.instance().theme.text, width=2))
        fonts.axis_title(plot.getPlotItem(), "left", "Power (dB)" if self._db else "Power",
                         ThemeHub.instance().theme.plot_foreground)

    def _redraw_average(self) -> None:
        pg = self._pg
        plot = self.average_plot
        self._forget(plot)
        legend = plot.getPlotItem().legend
        if legend is not None:
            legend.clear()
        for t in sorted(self._by_type):
            arr = self._data[self._by_type[t]]
            mean = self._scale(np.mean(arr, axis=0))
            colour = self._colour(t)
            if arr.shape[0] > 1:
                std = np.std(self._scale(arr), axis=0)
                band = QColor(colour)
                band.setAlpha(40)
                top = self._plot_curve(plot, mean + std, pen=pg.mkPen(color=band, width=1))
                bottom = self._plot_curve(plot, mean - std, pen=pg.mkPen(color=band, width=1))
                fill = pg.FillBetweenItem(top, bottom, brush=pg.mkBrush(band))
                plot.addItem(fill)
                self._drawn.setdefault(id(plot), []).append(fill)
            self._plot_curve(plot, mean, pen=pg.mkPen(color=colour, width=2), name=t)
        fonts.axis_title(plot.getPlotItem(), "left", "Power (dB)" if self._db else "Power",
                         ThemeHub.instance().theme.plot_foreground)


__all__ = ["PsdWindow"]
