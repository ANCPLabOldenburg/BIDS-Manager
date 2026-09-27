"""The power-spectral-density viewer, shared by everything that has a signal.

It started inside the MEG/EEG viewer and is now opened from four places: that
viewer, the Converter's per-row Properties panel, the Editor's physio plot,
and the tests. A class used by four modules is not a private detail of one of
them, so it lives here, along with the two palette-token colour helpers it
shares with the viewer.

What it draws is the same in every case, because a PSD is the same question
in every case: how much of this signal sits at each frequency. The caller
supplies frequencies, one power row per channel, and the names and kinds to
group and colour them by. A physio recording has no channel TYPES in the MEG
sense, so it passes the column names as the kinds and its own colour
function; nothing here needs to know which it is looking at.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import CUR

# channel-type -> palette token, so trace colours follow the theme.
TYPE_TOKENS: dict[str, str] = {
    "mag": "accent", "grad": "success", "eeg": "purple", "seeg": "purple",
    "ecog": "purple", "eog": "teal", "ecg": "error", "emg": "warning",
    "stim": "warning", "ref_meg": "dim", "misc": "dim", "bio": "teal",
    "resp": "teal", "dbs": "purple",
}

# Distinct colours (palette tokens) for event IDs / PSD curves.
SERIES_TOKENS: tuple[str, ...] = (
    "accent", "success", "purple", "teal", "warning", "error", "dim",
)


def series_color(index: int) -> str:
    """The ``index``-th distinct colour, cycling. Theme-aware."""
    pal = CUR()
    token = SERIES_TOKENS[index % len(SERIES_TOKENS)]
    return pal.get(token, pal.get("text", "#888888"))


#: Colours somebody chose for a channel type, keyed by type, as hex. Empty
#: means every type takes its theme token, which is the shipped scheme.
#:
#: A module-level cache rather than a settings read per curve: this is called
#: once for every channel drawn, on every redraw, and a three-hundred-channel
#: MEG window redraws as the pointer moves.
_TYPE_OVERRIDES: dict[str, str] = {}
_TYPE_OVERRIDES_LOADED = False


def default_type_color(ch_type: str) -> str:
    """The SHIPPED colour for a kind, ignoring anything somebody chose.

    Theme-aware, because the shipped scheme is palette tokens rather than
    literals: ``mag`` is the accent colour and ``grad`` the success colour,
    so they stay distinguishable and both stay legible in either theme.

    A type the map does not name gets a token derived FROM ITS NAME rather
    than the one grey they all used to share. A real MEG file carries
    ``ias`` and ``syst`` alongside ``misc``, and three different kinds of
    channel drawn in one colour is three kinds nobody can tell apart. The
    derivation is a character sum, not :func:`hash`, because Python
    randomises string hashing per process and a colour that changed every
    time the app started would be worse than a collision.
    """
    pal = CUR()
    token = TYPE_TOKENS.get(ch_type)
    if token is None:
        name = str(ch_type)
        token = SERIES_TOKENS[
            sum(ord(c) for c in name) % len(SERIES_TOKENS)
        ] if name else "dim"
    return pal.get(token, pal.get("text", "#888888"))


def type_colors() -> dict[str, str]:
    """The overrides currently in force. A copy, so callers cannot edit it."""
    _load_type_overrides()
    return dict(_TYPE_OVERRIDES)


def set_type_colors(mapping: Optional[dict] = None) -> None:
    """Install per-type colours. ``None`` or empty restores the defaults."""
    global _TYPE_OVERRIDES, _TYPE_OVERRIDES_LOADED
    _TYPE_OVERRIDES = {
        str(k): str(v) for k, v in dict(mapping or {}).items() if v
    }
    _TYPE_OVERRIDES_LOADED = True


def _load_type_overrides() -> None:
    global _TYPE_OVERRIDES_LOADED
    if _TYPE_OVERRIDES_LOADED:
        return
    _TYPE_OVERRIDES_LOADED = True
    try:
        from ..app_settings import AppSettings

        set_type_colors(AppSettings.load().trace_type_colors)
    except Exception:  # noqa: BLE001 - a preference is not worth a crash
        pass


def type_color(ch_type: str) -> str:
    """The colour a channel KIND is drawn in: chosen, else shipped."""
    _load_type_overrides()
    chosen = _TYPE_OVERRIDES.get(ch_type)
    return chosen if chosen else default_type_color(ch_type)


class PsdDialog(QDialog):
    """Power Spectral Density viewer.

    Two tabs:
    * **Per channel** - every channel of the selected type overlaid (thin),
      plus the type mean (bold); a "Highlight" picker isolates one channel.
    * **Average** - the per-type mean with a +/-1 std shaded band + legend.

    Both plots are fully interactive (pyqtgraph: drag-pan, wheel/right-drag
    zoom, auto-range button) and carry a crosshair with a frequency / power
    readout, mirroring the useful parts of MNE's spectrum plot.

    ``color_for`` maps a kind to a colour. It defaults to the MEG/EEG
    channel-type map; a caller whose kinds are not channel types (physio
    columns, say) passes its own rather than having every one of them come
    back as the same "unknown" grey.
    """

    def __init__(
        self,
        result: dict,
        *,
        parent: Optional[QWidget] = None,
        color_for: Optional[Callable[[str], str]] = None,
        title: str = "Power spectral density",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName("pane-dark")
        self.resize(900, 560)
        import pyqtgraph as pg

        self._pg = pg
        self._color_for = color_for or type_color
        self._freqs = np.asarray(result["freqs"])
        self._data = np.asarray(result["data"])          # (n_ch, n_freqs)
        self._ch_names: List[str] = list(result["ch_names"])
        self._ch_types: List[str] = list(result["ch_types"])
        # Defensive clamp: only index rows that actually exist in ``data``
        # (compute_psd returns a channel subset, so a caller passing the full
        # name/type lists must never drive an out-of-range index).
        n = min(self._data.shape[0], len(self._ch_names), len(self._ch_types))
        self._by_type: dict[str, List[int]] = {}
        for i in range(n):
            self._by_type.setdefault(self._ch_types[i], []).append(i)
        self._db = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Shared top bar.
        top = QHBoxLayout()
        self._db_chk = QCheckBox("dB (10·log10)")
        self._db_chk.setChecked(True)
        self._db_chk.toggled.connect(self._on_db_toggled)
        top.addWidget(self._db_chk)
        top.addStretch(1)
        hint = QLabel("Drag to pan · wheel / right-drag to zoom · click ⟲ to reset")
        hint.setObjectName("sidecar-footer-summary")
        top.addWidget(hint)
        outer.addLayout(top)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)
        self._build_channel_tab()
        self._build_average_tab()

        self._redraw_channel()
        self._redraw_average()

    # ---- tab construction -------------------------------------------------
    def _build_channel_tab(self) -> None:
        pg = self._pg
        page = QWidget()
        page.setObjectName("pane-dark")
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        for t in sorted(self._by_type):
            self._type_combo.addItem(f"{t}  ({len(self._by_type[t])})", userData=t)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        ctrl.addWidget(self._type_combo)
        ctrl.addWidget(QLabel("Highlight:"))
        self._hi_combo = QComboBox()
        self._hi_combo.currentIndexChanged.connect(self._redraw_channel)
        ctrl.addWidget(self._hi_combo, 1)
        v.addLayout(ctrl)

        self._ch_plot = pg.PlotWidget()
        self._ch_plot.showGrid(x=True, y=True, alpha=0.15)
        self._ch_plot.setLabel("bottom", "Frequency", units="Hz")
        v.addWidget(self._ch_plot, 1)
        self._ch_readout = QLabel("")
        self._ch_readout.setObjectName("sidecar-footer-summary")
        v.addWidget(self._ch_readout)
        self._theme_plot(self._ch_plot)
        self._ch_cross = self._wire_crosshair(self._ch_plot, self._ch_readout)
        self._tabs.addTab(page, "Per channel")
        self._refresh_highlight_combo()

    def _build_average_tab(self) -> None:
        pg = self._pg
        page = QWidget()
        page.setObjectName("pane-dark")
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        self._avg_plot = pg.PlotWidget()
        self._avg_plot.showGrid(x=True, y=True, alpha=0.15)
        self._avg_plot.setLabel("bottom", "Frequency", units="Hz")
        self._avg_plot.addLegend(offset=(-10, 10))
        v.addWidget(self._avg_plot, 1)
        self._avg_readout = QLabel("")
        self._avg_readout.setObjectName("sidecar-footer-summary")
        v.addWidget(self._avg_readout)
        self._theme_plot(self._avg_plot)
        self._avg_cross = self._wire_crosshair(self._avg_plot, self._avg_readout)
        self._tabs.addTab(page, "Average (per type)")

    # ---- helpers ----------------------------------------------------------
    def _scale(self, arr):
        if self._db:
            return 10.0 * np.log10(np.maximum(arr, 1e-30))
        return arr

    def _ylabel(self) -> str:
        return "Power (dB)" if self._db else "Power"

    def _theme_plot(self, plot) -> None:
        pg = self._pg
        bg = self.palette().color(QPalette.ColorRole.Base)
        fg = self.palette().color(QPalette.ColorRole.Text)
        plot.setBackground(bg)
        for axis in ("left", "bottom"):
            ax = plot.getPlotItem().getAxis(axis)
            ax.setPen(pg.mkPen(color=fg))
            ax.setTextPen(pg.mkPen(color=fg))

    def _wire_crosshair(self, plot, readout: QLabel):
        pg = self._pg
        pen = pg.mkPen(color=self.palette().color(QPalette.ColorRole.Text), width=1,
                       style=Qt.PenStyle.DashLine)
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        vline.setZValue(10)
        hline.setZValue(10)
        plot.addItem(vline, ignoreBounds=True)
        plot.addItem(hline, ignoreBounds=True)

        def _moved(evt):
            pos = evt[0]
            vb = plot.getPlotItem().vb
            if not plot.sceneBoundingRect().contains(pos):
                return
            pt = vb.mapSceneToView(pos)
            vline.setPos(pt.x())
            hline.setPos(pt.y())
            readout.setText(
                f"{pt.x():.2f} Hz   ·   {pt.y():.2f} {self._ylabel()}"
            )

        proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=30, slot=_moved)
        return {"vline": vline, "hline": hline, "proxy": proxy}

    def _refresh_highlight_combo(self) -> None:
        t = self._type_combo.currentData()
        self._hi_combo.blockSignals(True)
        self._hi_combo.clear()
        self._hi_combo.addItem("(none)", userData=-1)
        for idx in self._by_type.get(t, []):
            self._hi_combo.addItem(self._ch_names[idx], userData=idx)
        self._hi_combo.blockSignals(False)

    # ---- draw -------------------------------------------------------------
    def _on_db_toggled(self, checked: bool) -> None:
        self._db = checked
        self._redraw_channel()
        self._redraw_average()

    def _on_type_changed(self) -> None:
        self._refresh_highlight_combo()
        self._redraw_channel()

    def _redraw_channel(self) -> None:
        pg = self._pg
        plot = self._ch_plot
        plot.clear()
        plot.addItem(self._ch_cross["vline"], ignoreBounds=True)
        plot.addItem(self._ch_cross["hline"], ignoreBounds=True)
        t = self._type_combo.currentData()
        idxs = self._by_type.get(t, [])
        if not idxs:
            return
        base = QColor(self._color_for(t))
        thin = QColor(base)
        thin.setAlpha(60)
        thin_pen = pg.mkPen(color=thin, width=1)
        for i in idxs:
            plot.plot(self._freqs, self._scale(self._data[i]), pen=thin_pen)
        # Mean (bold).
        mean = np.mean(self._data[idxs], axis=0)
        plot.plot(self._freqs, self._scale(mean),
                  pen=pg.mkPen(color=base, width=2), name="mean")
        # Highlighted channel.
        hi = self._hi_combo.currentData()
        if hi is not None and hi >= 0:
            hi_color = self.palette().color(QPalette.ColorRole.Text)
            plot.plot(self._freqs, self._scale(self._data[hi]),
                      pen=pg.mkPen(color=hi_color, width=2))
        plot.setLabel("left", self._ylabel())

    def _redraw_average(self) -> None:
        pg = self._pg
        plot = self._avg_plot
        plot.clear()
        legend = plot.getPlotItem().legend
        if legend is not None:
            legend.clear()
        plot.addItem(self._avg_cross["vline"], ignoreBounds=True)
        plot.addItem(self._avg_cross["hline"], ignoreBounds=True)
        for t in sorted(self._by_type):
            idxs = self._by_type[t]
            arr = self._data[idxs]
            mean = self._scale(np.mean(arr, axis=0))
            color = QColor(self._color_for(t))
            if arr.shape[0] > 1:
                std = np.std(self._scale(arr), axis=0)
                band = QColor(color)
                band.setAlpha(40)
                top = plot.plot(self._freqs, mean + std,
                                pen=pg.mkPen(color=band, width=1))
                bot = plot.plot(self._freqs, mean - std,
                                pen=pg.mkPen(color=band, width=1))
                fill = pg.FillBetweenItem(top, bot, brush=pg.mkBrush(band))
                plot.addItem(fill)
            plot.plot(self._freqs, mean, pen=pg.mkPen(color=color, width=2), name=t)
        plot.setLabel("left", self._ylabel())


__all__ = [
    "PsdDialog",
    "SERIES_TOKENS",
    "TYPE_TOKENS",
    "default_type_color",
    "series_color",
    "set_type_colors",
    "type_color",
    "type_colors",
]
