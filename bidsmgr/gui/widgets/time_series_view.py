"""The interactive multi-channel time-series viewer, shared by every signal.

A restyled, annotation-free port of the MEEGqc ``qc_viewer`` widget
(pyqtgraph). It started inside the MEG/EEG pane and is now driven by two of
them: that pane, and the physio viewer. A class used by more than one module
is not a private detail of either, so it lives here.

**Everything it needs is an ``mne.io.Raw``**, and that is the whole reason
the physio viewer can use it. A ``*_physio.tsv.gz`` is not an MNE format,
but it IS channels, a sampling rate and samples, which is what a
``RawArray`` is made of. Building one costs a few lines in
``physio_viewer`` and buys the channel picker, the navigation, the filters,
the resample, the spectrum and the events overlay, all already written and
already tested, instead of a second implementation of each that drifts.

What resets per recording, and what does not
--------------------------------------------
:meth:`load_raw` resets everything MEASURED IN THE RECORDING'S OWN TERMS
(the time window, the scroll position, the channel offset, the amplitude
scale) and keeps everything that is a standing preference (the dark-plot
override, the event colour and width).

The distinction is not cosmetic. A ten-second window carried from a
ten-second recording onto a forty-minute one opens the new file zoomed into
its first 0.4 percent, which reads as "the viewer loaded it wrong". A window
length is only meaningful relative to a duration, so it cannot persist
across recordings; a colour means the same thing everywhere, so it can.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .. import icons
from .flow_layout import flow
from .primitives import ElidedLabel
from .recording_formats import full_ext
from .psd_dialog import PsdDialog, series_color, set_type_colors, type_color

log = logging.getLogger(__name__)


def _nan_mean(values) -> float:
    """``np.nanmean`` that survives an all-gap channel."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else 0.0


def _nan_ptp(values) -> float:
    """Peak-to-peak ignoring gaps; 0 when there is nothing to measure."""
    finite = values[np.isfinite(values)]
    return float(finite.max() - finite.min()) if finite.size else 0.0

#: Seconds of signal a recording opens showing. Ten is enough to see the
#: shape of a cardiac or respiratory cycle and short enough that a long
#: recording does not draw a million points before the user has asked for
#: anything.
_DEFAULT_WINDOW_S = 10.0

#: Channels drawn at once before the scrollbar takes over.
_DEFAULT_VISIBLE = 20

#: Decimal places on every numeric field that takes a frequency or a gain.
#: Three, because one is not enough to type a 0.01 Hz high-pass and a
#: respiratory drift sits below 0.1 Hz. Reported from use: "impossible to
#: put a filter of 0.01".
_FILTER_DECIMALS = 3

#: Samples a single redraw may read for the sake of filter padding, across
#: all visible channels. One physio channel can afford the 330 seconds
#: either side a 0.01 Hz high-pass wants; three hundred MEG channels cannot,
#: and the budget is what lets the same code serve both.
_FILTER_SAMPLE_BUDGET = 4_000_000

#: Trace width when only a few traces are on screen. A hairline reads as a
#: scratch on a modern display and the viewers people compare this to draw
#: thicker, so a single physio channel gets two pixels.
_THICK_LINE_WIDTH = 2

#: Above this many traces NOTHING wider than one pixel is drawn, whatever is
#: asked for, and it is not a matter of taste. Qt strokes a cosmetic (one
#: pixel) pen through a fast path and has to stroke anything wider properly.
#: Measured on a 323-channel MEG recording, painting the pane:
#:
#: ===== ======= =======
#: shown width 1 width 2
#: ===== ======= =======
#: 1     4.1 ms  12.3 ms
#: 2     4.0 ms   9.4 ms
#: 8     11.3 ms 79.8 ms
#: 20    12.9 ms 88.2 ms
#: 323   145 ms  809 ms
#: ===== ======= =======
#:
#: A drag redraws continuously, so above two traces the wider pen is the
#: difference between a view that follows the cursor and one that does not.
#: Reducing the point count does not rescue it: peak decimation cut a
#: twenty-channel window from 200,000 points to 69,000 and width two only
#: came down from 115 ms to 90, because the cost is the stroking.
_THICK_LINE_MAX_TRACES = 2

#: What a width of zero means: decide it from what is on screen. A width the
#: user picked in the Line popup is honoured whatever it costs, because at
#: that point the cost is theirs to choose.
_LINE_WIDTH_AUTO = 0


def _peak_decimate(times, values, pixels: int):
    """Reduce a trace to about two points per pixel, KEEPING THE EXTREMES.

    A pane is on the order of a thousand pixels wide. Asking Qt to stroke a
    forty-minute recording sample by sample is a million points into a
    thousand columns, a thousand of them per column, and all but two of
    those thousand land on a pixel another one already covered: the work is
    real and the result is identical. That is what made Fit all freeze.

    Taking every n-th sample would be wrong, not just lossy: the peak of an
    R wave or a trigger one sample wide falls between the samples kept and
    the feature DISAPPEARS, which is worse than slow because it is quietly
    incorrect. So each column keeps its MINIMUM and its MAXIMUM, which is
    the standard answer and preserves the envelope exactly: what is drawn
    covers the same pixels the full trace would have.

    A gap stays a gap. A column is NaN only when every sample in it is,
    because a column holding one dropped sample among a thousand good ones
    still has a range worth drawing, and propagating the NaN would open a
    hole the recording does not have.
    """
    n = int(np.asarray(values).size)
    target = max(256, int(pixels) * 2)
    if n <= target:
        return times, values
    buckets = target // 2
    usable = (n // buckets) * buckets
    if usable < buckets * 2:
        return times, values

    block = np.asarray(values[:usable], dtype=float).reshape(buckets, -1)
    finite = np.isfinite(block)
    has_any = finite.any(axis=1)
    # +inf / -inf rather than nanmin / nanmax: same answer, no all-NaN
    # RuntimeWarning to suppress, and one pass instead of two.
    lo = np.where(finite, block, np.inf).min(axis=1)
    hi = np.where(finite, block, -np.inf).max(axis=1)
    lo = np.where(has_any, lo, np.nan)
    hi = np.where(has_any, hi, np.nan)

    t = np.asarray(times[:usable], dtype=float).reshape(buckets, -1)
    out_t = np.empty(buckets * 2, dtype=float)
    out_y = np.empty(buckets * 2, dtype=float)
    out_t[0::2] = t[:, 0]
    out_t[1::2] = t[:, -1]
    out_y[0::2] = lo
    out_y[1::2] = hi
    # Whatever the reshape could not cover, at most one bucket's worth, so
    # the trace still reaches the right-hand edge.
    if usable < n:
        out_t = np.concatenate([out_t, np.asarray(times[usable:], dtype=float)])
        out_y = np.concatenate([out_y, np.asarray(values[usable:], dtype=float)])
    return out_t, out_y


def _events_sibling(path) -> Optional[Path]:
    """Return the sibling BIDS ``*_events.tsv`` for *path*, if present."""
    p = Path(path)
    name = p.name
    ext = full_ext(p)
    if ext and name.lower().endswith(ext):
        stem = name[: -len(ext)]
    else:
        stem = p.stem
    base = stem.rsplit("_", 1)[0] if "_" in stem else stem
    for suffix in ("_events.tsv", "_events.tsv.gz"):
        cand = p.parent / f"{base}{suffix}"
        if cand.exists():
            return cand
    return None


def _read_events_tsv(path) -> List[tuple]:
    """Parse a BIDS ``events.tsv`` into ``[(onset_s, label), ...]``."""
    import csv
    import gzip

    out: List[tuple] = []
    is_gz = str(path).lower().endswith(".gz")
    try:
        opener = (
            gzip.open(path, "rt", encoding="utf-8", newline="")
            if is_gz
            else open(path, "r", encoding="utf-8", newline="")
        )
        with opener as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                onset = row.get("onset")
                if onset is None:
                    continue
                try:
                    t = float(onset)
                except (TypeError, ValueError):
                    continue
                label = (
                    row.get("trial_type")
                    or row.get("value")
                    or row.get("event_type")
                    or ""
                )
                out.append((t, str(label)))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not read events.tsv %s: %s", path, exc)
        return []
    return out


class TimeSeriesView(QWidget):
    """Interactive multi-channel EEG/MEG trace viewer (pyqtgraph)."""

    status_message = pyqtSignal(str)
    loading_changed = pyqtSignal(bool, str)
    close_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")

        self._raw = None
        self._sfreq = 1000.0
        self._ch_names: List[str] = []
        self._ch_types: List[str] = []
        self._available_ch_types: List[str] = []
        self._is_ctf = False
        self._duration = 0.0
        self._n_samples = 0

        self._visible_channels = _DEFAULT_VISIBLE
        self._channel_offset = 0
        self._time_start = 0.0
        self._time_window = _DEFAULT_WINDOW_S
        self._scale_factor = 1.0
        self._active_ch_type = "all"
        self._selected_channels: Optional[Set[str]] = None
        self._normalize = False
        self._dark_plot = False
        # Where the recording has no samples. MNE cannot carry NaN through a
        # filter or an FFT, so the RawArray holds zeros and the mask says
        # where they are: the filter sees a continuous signal and the DRAWING
        # shows the gap. Without it, 28 percent of a real ECG was being drawn
        # as a line at zero, which on a trace centred near 2050 is a spike to
        # the floor of the plot. Reported as "artifacts".
        self._gaps: Optional[np.ndarray] = None
        # How the traces are drawn. Automatic by default, which means two
        # pixels for a physio channel and one for a wall of MEG: see
        # ``_THICK_LINE_MAX_TRACES`` for the measurement that forces it.
        self._line_width = _LINE_WIDTH_AUTO
        self._line_color: Optional[str] = None   # None = colour by channel type
        self._restore_line_style()

        self._display_indices: List[int] = []
        self._shown_ch_info: list = []
        self._overlay_items: list = []

        self._current_filter = None
        self._notch_freq = None
        self._filter_pad_warned = None
        self._filter_short_warned = None
        self._current_filepath: Optional[str] = None
        self._current_root: Optional[Path] = None

        # Events (BIDS-native).
        self._show_events = False
        self._event_source = "auto"
        self._event_line_width = 2
        self._event_color_override: Optional[QColor] = None
        self._tsv_events: List[tuple] = []
        self._stim_events: List[tuple] = []
        self._stim_channels: List[str] = []

        self._resample_worker = None
        self._psd_worker = None

        import pyqtgraph as pg

        self._pg = pg
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(2)

        self._toolbar_widget = self._build_toolbar()
        self._toolbar_toggle = self._make_section_toggle(
            "Display controls", self._toolbar_widget, expanded=True
        )
        main.addWidget(self._toolbar_toggle)
        main.addWidget(self._toolbar_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._label_area = QWidget()
        self._label_layout = QVBoxLayout(self._label_area)
        self._label_layout.setContentsMargins(2, 0, 2, 0)
        self._label_layout.setSpacing(0)
        label_scroll = QScrollArea()
        label_scroll.setWidget(self._label_area)
        label_scroll.setWidgetResizable(True)
        label_scroll.setFixedWidth(68)
        label_scroll.setObjectName("viewer-label-strip")
        label_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._plot_widget = self._pg.PlotWidget()
        self._plot_widget.setMinimumWidth(60)
        self._plot_widget.showGrid(x=True, y=False, alpha=0.15)
        self._plot_widget.setLabel("bottom", "Time", units="s")
        # pyqtgraph's own panning is OFF, and that is deliberate.
        #
        # Letting the ViewBox pan means dragging moves the CAMERA over the
        # ten seconds that were fetched, so the recording appears to end at
        # the edge of the window and everything beyond it is blank. What a
        # reader means by dragging a trace is "show me further along",
        # which is what the slider does, so the drag is translated into the
        # same thing: it moves the window through the recording and the
        # next stretch is read from disk.
        self._plot_widget.setMouseEnabled(x=False, y=False)
        self._install_drag_to_scrub()
        self._plot_widget.getPlotItem().getAxis("left").setWidth(0)
        self._plot_widget.getPlotItem().getAxis("left").setStyle(
            showValues=False
        )
        self._plot_widget.wheelEvent = self._on_plot_wheel
        # The names are placed from the view's geometry, so they have to
        # follow it when the pane is resized or a splitter is dragged. Cheap:
        # it moves existing labels and reads nothing.
        self._channel_labels: list = []
        self._plot_widget.getPlotItem().getViewBox().sigResized.connect(
            self._position_labels
        )
        self._hover_proxy = self._pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_mouse_moved,
        )

        splitter.addWidget(label_scroll)
        splitter.addWidget(self._plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._chan_scroll = QScrollBar(Qt.Orientation.Vertical)
        self._chan_scroll.setToolTip("Scroll channels")
        self._chan_scroll.valueChanged.connect(self._on_channel_scroll)

        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(0)
        plot_row.addWidget(splitter, 1)
        plot_row.addWidget(self._chan_scroll)
        plot_container = QWidget()
        plot_container.setLayout(plot_row)
        main.addWidget(plot_container, 1)

        main.addWidget(self._build_navigation())

        self._events_widget = self._build_events_row()
        self._events_toggle = self._make_section_toggle(
            "Events", self._events_widget, expanded=False
        )
        main.addWidget(self._events_toggle)
        main.addWidget(self._events_widget)

        self._apply_plot_theme()

    def _make_section_toggle(
        self, title: str, widget: QWidget, *, expanded: bool
    ) -> QPushButton:
        btn = QPushButton(("▼ " if expanded else "▶ ") + title)
        btn.setCheckable(True)
        btn.setChecked(expanded)
        btn.setObjectName("viewer-section-toggle")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        widget.setVisible(expanded)

        def _toggle(checked: bool, w=widget, b=btn, t=title) -> None:
            w.setVisible(checked)
            b.setText(("▼ " if checked else "▶ ") + t)

        btn.clicked.connect(_toggle)
        return btn

    def _build_toolbar(self) -> QWidget:
        """Two rows of controls that WRAP as the pane narrows.

        They used to scroll sideways instead. That kept the pane shrinkable,
        which was the point, but it put every control past the first
        screenful behind a horizontal scrollbar nobody looks for. Wrapping
        does the same job better: nothing goes out of reach, the bar grows
        taller instead, and the minimum width becomes the widest single
        control rather than the widest row.
        """
        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # Row 1 - display
        row1 = QFrame()
        row1.setObjectName("toolbar")
        l1 = flow(row1, h_spacing=6, v_spacing=4)
        l1.setContentsMargins(10, 4, 10, 4)

        self._lbl_type = QLabel("Type:")
        l1.addWidget(self._lbl_type)
        self.cmb_ch_type = QComboBox()
        self.cmb_ch_type.addItem("all")
        self.cmb_ch_type.currentTextChanged.connect(self._on_ch_type_changed)
        l1.addWidget(self.cmb_ch_type)

        self.btn_select = QPushButton("  Channels")
        self.btn_select.setObjectName("tb-btn")
        icons.apply_button(self.btn_select, "channels")
        self.btn_select.setToolTip("Pick specific channels to display")
        self.btn_select.clicked.connect(self._open_channel_selector)
        l1.addWidget(self.btn_select)

        self._lbl_count = QLabel("Count:")
        l1.addWidget(self._lbl_count)
        self.spn_n = QSpinBox()
        self.spn_n.setRange(1, 500)
        self.spn_n.setValue(self._visible_channels)
        self.spn_n.valueChanged.connect(self._on_n_changed)
        l1.addWidget(self.spn_n)

        l1.addWidget(QLabel("Scale:"))
        self.spn_scale = QDoubleSpinBox()
        self.spn_scale.setRange(0.01, 1000.0)
        self.spn_scale.setValue(1.0)
        self.spn_scale.setSingleStep(0.1)
        self.spn_scale.setDecimals(_FILTER_DECIMALS)
        self.spn_scale.valueChanged.connect(self._on_scale_changed)
        l1.addWidget(self.spn_scale)

        l1.addWidget(QLabel("Window (s):"))
        self.spn_window = QDoubleSpinBox()
        self.spn_window.setRange(0.1, 3600.0)
        self.spn_window.setValue(self._time_window)
        self.spn_window.setSingleStep(1.0)
        self.spn_window.valueChanged.connect(self._on_window_changed)
        l1.addWidget(self.spn_window)

        self.chk_dark = QCheckBox("Dark plot")
        self.chk_dark.toggled.connect(self._on_dark_toggled)
        l1.addWidget(self.chk_dark)

        self.chk_norm = QCheckBox("Normalize")
        self.chk_norm.setToolTip(
            "Normalise each channel independently to prevent overlap.\n"
            "When off, raw signals share a per-type scale (may overlap)."
        )
        self.chk_norm.toggled.connect(self._on_norm_toggled)
        l1.addWidget(self.chk_norm)

        self.btn_reset = QPushButton("  Reset view")
        self.btn_reset.setObjectName("tb-btn")
        icons.apply_button(self.btn_reset, "reset_view")
        self.btn_reset.clicked.connect(self._reset_view)
        l1.addWidget(self.btn_reset)

        self.btn_close = QPushButton("  Close")
        self.btn_close.setObjectName("tb-btn")
        icons.apply_button(self.btn_close, "close_data")
        self.btn_close.setToolTip("Close the signal and return to the metadata view")
        self.btn_close.clicked.connect(self.close_requested.emit)
        l1.addWidget(self.btn_close)

        l1.addStretch(1)
        cl.addWidget(row1)

        # Row 2 - processing
        row2 = QFrame()
        row2.setObjectName("toolbar")
        l2 = flow(row2, h_spacing=6, v_spacing=4)
        l2.setContentsMargins(10, 4, 10, 4)

        l2.addWidget(QLabel("HP (Hz):"))
        self.spn_hp = QDoubleSpinBox()
        self.spn_hp.setRange(0.0, 500.0)
        # THREE decimals, not one. A high-pass at 0.01 Hz is the band a
        # respiratory drift lives in, and at one decimal the smallest value
        # that could be typed was 0.1 Hz, which removes the respiration
        # along with the drift.
        self.spn_hp.setDecimals(_FILTER_DECIMALS)
        self.spn_hp.setSingleStep(0.1)
        self.spn_hp.setSpecialValueText("Off")
        l2.addWidget(self.spn_hp)

        l2.addWidget(QLabel("LP (Hz):"))
        self.spn_lp = QDoubleSpinBox()
        self.spn_lp.setRange(0.0, 5000.0)
        self.spn_lp.setDecimals(_FILTER_DECIMALS)
        self.spn_lp.setSpecialValueText("Off")
        l2.addWidget(self.spn_lp)

        l2.addWidget(QLabel("Notch (Hz):"))
        self.spn_notch = QDoubleSpinBox()
        self.spn_notch.setRange(0.0, 1000.0)
        self.spn_notch.setDecimals(_FILTER_DECIMALS)
        self.spn_notch.setSpecialValueText("Off")
        l2.addWidget(self.spn_notch)

        self.btn_filter = QPushButton("  Apply filter")
        self.btn_filter.setObjectName("tb-btn")
        icons.apply_button(self.btn_filter, "filter")
        self.btn_filter.clicked.connect(self._apply_filters)
        l2.addWidget(self.btn_filter)

        self.btn_filter_reset = QPushButton("Reset filters")
        self.btn_filter_reset.setObjectName("tb-btn")
        self.btn_filter_reset.clicked.connect(self._reset_filters)
        l2.addWidget(self.btn_filter_reset)

        l2.addWidget(QLabel("Resample (Hz):"))
        self.spn_resample = QDoubleSpinBox()
        self.spn_resample.setRange(0.0, 10000.0)
        self.spn_resample.setDecimals(0)
        self.spn_resample.setSpecialValueText("Off")
        l2.addWidget(self.spn_resample)

        self.btn_resample = QPushButton("  Resample")
        self.btn_resample.setObjectName("tb-btn")
        icons.apply_button(self.btn_resample, "resample")
        self.btn_resample.clicked.connect(self._apply_resample)
        l2.addWidget(self.btn_resample)

        self.btn_psd = QPushButton("  PSD")
        self.btn_psd.setObjectName("tb-btn")
        icons.apply_button(self.btn_psd, "psd")
        self.btn_psd.setToolTip("Power spectral density (interactive, in-app)")
        self.btn_psd.clicked.connect(self._show_psd)
        l2.addWidget(self.btn_psd)

        self.btn_line = QPushButton("  Line")
        self.btn_line.setObjectName("tb-btn")
        self.btn_line.setToolTip(
            "How the traces are drawn: thickness, and whether they take "
            "their channel type's colour or one you pick."
        )
        self.btn_line.clicked.connect(self._open_line_style)
        l2.addWidget(self.btn_line)

        self.btn_fit = QPushButton("  Fit all")
        self.btn_fit.setObjectName("tb-btn")
        self.btn_fit.setToolTip(
            "Put the whole recording in the window at once, instead of "
            "winding the window length up a step at a time."
        )
        self.btn_fit.clicked.connect(self._fit_all)
        # OFF unless the caller says the recording is small enough. On a
        # 300-channel MEG recording of forty minutes, the whole recording is
        # a quarter of a billion samples to read, filter and stroke, so the
        # button would be a freeze with a label on it. Physio asks for it;
        # MEG and EEG do not. See ``enable_fit_all``.
        self.btn_fit.setVisible(False)
        l2.addWidget(self.btn_fit)

        l2.addStretch(1)
        cl.addWidget(row2)

        # No scroll area, and no maximum height. Both existed to stop the
        # toolbar forcing a wide minimum on the pane, which the wrapping
        # rows now do properly; a height cap on a bar whose rows reflow
        # would clip the rows it grows.
        container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum,
        )
        return container

    def _build_navigation(self) -> QWidget:
        nav = QFrame()
        nav.setObjectName("toolbar")
        lay = QHBoxLayout(nav)
        lay.setContentsMargins(10, 2, 10, 2)
        lay.setSpacing(4)

        self.btn_start = QPushButton("|<")
        self.btn_start.setObjectName("tb-btn")
        self.btn_start.setFixedWidth(40)
        self.btn_start.clicked.connect(lambda: self._navigate("start"))
        self.btn_prev = QPushButton("<")
        self.btn_prev.setObjectName("tb-btn")
        self.btn_prev.setFixedWidth(34)
        self.btn_prev.clicked.connect(lambda: self._navigate("prev"))
        self.btn_next = QPushButton(">")
        self.btn_next.setObjectName("tb-btn")
        self.btn_next.setFixedWidth(34)
        self.btn_next.clicked.connect(lambda: self._navigate("next"))
        self.btn_end = QPushButton(">|")
        self.btn_end.setObjectName("tb-btn")
        self.btn_end.setFixedWidth(40)
        self.btn_end.clicked.connect(lambda: self._navigate("end"))

        self.sld_time = QSlider(Qt.Orientation.Horizontal)
        self.sld_time.setRange(0, 1000)
        # A QSlider asks for a comfortable length; on a narrow pane there is
        # no comfortable length and asking for one is what stops the pane
        # narrowing at all.
        self.sld_time.setMinimumWidth(40)
        self.sld_time.valueChanged.connect(self._on_time_slider)
        # Elided: it carries a running "12.3 / 600.0 s", and a plain QLabel
        # reports its longest text as its minimum.
        self.lbl_time = ElidedLabel("0.0 / 0.0 s")
        self.lbl_time.setObjectName("sidecar-footer-summary")

        lay.addWidget(self.btn_start)
        lay.addWidget(self.btn_prev)
        lay.addWidget(self.sld_time, 1)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.btn_end)
        lay.addWidget(self.lbl_time)
        return nav

    def _build_events_row(self) -> QWidget:
        box = QFrame()
        box.setObjectName("toolbar")
        lay = flow(box, h_spacing=6, v_spacing=4)
        lay.setContentsMargins(10, 4, 10, 4)

        self.chk_events = QCheckBox("Show events")
        self.chk_events.setEnabled(False)
        self.chk_events.toggled.connect(self._on_events_toggled)
        lay.addWidget(self.chk_events)

        lay.addWidget(QLabel("Source:"))
        self.cmb_event_src = QComboBox()
        self.cmb_event_src.addItem("auto")
        self.cmb_event_src.setEnabled(False)
        self.cmb_event_src.currentTextChanged.connect(self._on_event_src_changed)
        lay.addWidget(self.cmb_event_src)

        lay.addWidget(QLabel("Width:"))
        self.spn_event_w = QSpinBox()
        self.spn_event_w.setRange(1, 10)
        self.spn_event_w.setValue(self._event_line_width)
        self.spn_event_w.setFixedWidth(54)
        self.spn_event_w.valueChanged.connect(self._on_event_width_changed)
        lay.addWidget(self.spn_event_w)

        self.btn_event_color = QPushButton("Color…")
        self.btn_event_color.setObjectName("tb-btn")
        self.btn_event_color.clicked.connect(self._pick_event_color)
        lay.addWidget(self.btn_event_color)
        self.btn_event_color_reset = QPushButton("Reset")
        self.btn_event_color_reset.setObjectName("tb-btn")
        self.btn_event_color_reset.clicked.connect(self._reset_event_color)
        lay.addWidget(self.btn_event_color_reset)

        lay.addStretch(1)
        return box

    # -------------------------------------------------------------- loading
    def set_current_filepath(self, path, root) -> None:
        self._current_filepath = str(path) if path else None
        self._current_root = root

    def load_raw(self, raw, gaps=None) -> None:
        """Accept a preloaded ``mne.io.Raw`` and render it.

        ``gaps`` is an optional boolean array shaped like the data,
        True where the recording has no sample. See ``_gaps``.
        """
        import mne

        self._raw = raw
        self._gaps = gaps
        info = raw.info
        self._sfreq = float(info["sfreq"])
        self._ch_names = list(raw.ch_names)
        self._n_samples = int(raw.n_times)
        self._duration = float(raw.times[-1]) if raw.n_times else 0.0
        self._ch_types = [
            mne.channel_type(info, i) for i in range(len(self._ch_names))
        ]
        self._time_start = 0.0
        self._channel_offset = 0
        self._current_filter = None
        self._notch_freq = None
        self._selected_channels = None
        self._available_ch_types = sorted(set(self._ch_types))
        self._is_ctf = (
            "mag" in self._available_ch_types
            and "grad" not in self._available_ch_types
            and (
                "ref_meg" in self._available_ch_types
                or getattr(raw, "compensation_grade", None) is not None
                or (
                    self._current_filepath
                    and full_ext(self._current_filepath) == ".ds"
                )
            )
        )

        # EVERY control measured in this recording's own terms goes back to
        # its default. A ten-second window carried over from a ten-second
        # file opens a forty-minute one zoomed into its first 0.4 percent,
        # which reads as the viewer having loaded it wrong, and was reported
        # as exactly that. A window length only means something relative to
        # a duration, so it cannot survive a change of recording.
        #
        # Standing preferences are NOT touched here: the dark-plot override,
        # the event colour and the event line width mean the same thing on
        # every file and are set deliberately.
        self._time_window = min(_DEFAULT_WINDOW_S, max(0.1, self._duration))
        self._scale_factor = 1.0
        self._visible_channels = min(_DEFAULT_VISIBLE, len(self._ch_names)) or 1
        self._normalize = False
        self._active_ch_type = "all"

        controls = (
            self.spn_window, self.spn_scale, self.spn_n, self.chk_norm,
            self.spn_hp, self.spn_lp, self.spn_notch,
        )
        for w in controls:
            w.blockSignals(True)
        self.spn_window.setMaximum(max(0.1, self._duration))
        self.spn_window.setValue(self._time_window)
        self.spn_scale.setValue(1.0)
        self.spn_n.setValue(self._visible_channels)
        self.chk_norm.setChecked(False)
        # The filter boxes too: a 40 Hz low-pass set for an EEG recording is
        # above the Nyquist of a 50 Hz respiratory belt, and carrying it over
        # would silently do nothing while claiming to filter.
        self.spn_hp.setValue(0.0)
        self.spn_lp.setValue(0.0)
        self.spn_notch.setValue(0.0)
        for w in controls:
            w.blockSignals(False)

        self._rebuild_ch_type_combo()
        self._sync_single_channel_controls()
        self._update_display_indices()
        self._update_channel_scrollbar()
        self._extract_events(raw)
        self._refresh_event_controls()
        self._apply_plot_theme()
        self._redraw()

        display = []
        for ct in self._available_ch_types:
            display.append("mag (axial grad)" if self._is_ctf and ct == "mag" else ct)
        self.status_message.emit(
            f"Loaded {len(self._ch_names)} channels, "
            f"{self._duration:.1f}s @ {self._sfreq:.0f} Hz | "
            f"Types: {', '.join(display)}"
        )

    def _rebuild_ch_type_combo(self) -> None:
        self.cmb_ch_type.blockSignals(True)
        self.cmb_ch_type.clear()
        self.cmb_ch_type.addItem("all")
        has_mag = "mag" in self._available_ch_types
        has_grad = "grad" in self._available_ch_types
        if has_mag and has_grad:
            self.cmb_ch_type.addItem("mag+grad")
        for ct in self._available_ch_types:
            label = "mag (axial grad)" if self._is_ctf and ct == "mag" else ct
            self.cmb_ch_type.addItem(label, userData=ct)
        self.cmb_ch_type.setCurrentText("all")
        self._active_ch_type = "all"
        self.cmb_ch_type.blockSignals(False)

    # --------------------------------------------------------------- events
    def _extract_events(self, raw) -> None:
        import mne

        self._stim_events = []
        self._stim_channels = []
        try:
            stim = [
                ch
                for ch, t in zip(raw.ch_names, self._ch_types)
                if t == "stim"
            ]
            self._stim_channels = stim
            if stim:
                try:
                    events = mne.find_events(
                        raw, stim_channel=stim, shortest_event=1, verbose=False
                    )
                except Exception:
                    events = np.empty((0, 3), dtype=int)
                for ev in events:
                    eid = int(ev[2])
                    if eid != 0:
                        self._stim_events.append((float(ev[0]) / self._sfreq, eid))
        except Exception:
            pass

        self._tsv_events = []
        if self._current_filepath:
            sib = _events_sibling(self._current_filepath)
            if sib is not None:
                self._tsv_events = _read_events_tsv(sib)

    def _refresh_event_controls(self) -> None:
        sources = []
        if self._tsv_events:
            sources.append("events.tsv")
        if self._stim_events:
            sources.append("stim")
        has_any = bool(sources)
        self.chk_events.blockSignals(True)
        self.chk_events.setChecked(False)
        self.chk_events.setEnabled(has_any)
        self.chk_events.blockSignals(False)
        self._show_events = False
        self.cmb_event_src.blockSignals(True)
        self.cmb_event_src.clear()
        self.cmb_event_src.addItem("auto")
        for s in sources:
            self.cmb_event_src.addItem(s)
        self.cmb_event_src.setEnabled(has_any)
        self.cmb_event_src.blockSignals(False)
        self._event_source = "auto"
        if has_any:
            n = len(self._tsv_events) + len(self._stim_events)
            self.chk_events.setToolTip(
                f"Overlay {n} event marker(s) from "
                f"{' + '.join(sources)}"
            )
        else:
            self.chk_events.setToolTip(
                "No events found (no sibling events.tsv and no stim channel)"
            )

    def _active_events(self) -> List[tuple]:
        """Return ``[(time, label), ...]`` for the chosen source."""
        src = self._event_source
        out: List[tuple] = []
        if src in ("auto", "events.tsv") and self._tsv_events:
            out.extend((float(t), str(lbl)) for t, lbl in self._tsv_events)
            if src == "auto":
                return out
        if src in ("auto", "stim") and self._stim_events:
            out.extend((float(t), str(eid)) for t, eid in self._stim_events)
        return out

    # ------------------------------------------------------------- display
    def _update_display_indices(self) -> None:
        if self._active_ch_type == "all":
            indices = list(range(len(self._ch_names)))
        elif self._active_ch_type == "mag+grad":
            indices = [
                i for i, t in enumerate(self._ch_types) if t in ("mag", "grad")
            ]
        else:
            indices = [
                i
                for i, t in enumerate(self._ch_types)
                if t == self._active_ch_type
            ]
        if self._selected_channels is not None:
            indices = [
                i for i in indices if self._ch_names[i] in self._selected_channels
            ]
        self._display_indices = indices
        self._update_channel_scrollbar()

    def _update_channel_scrollbar(self) -> None:
        n = len(self._display_indices)
        visible = min(self._visible_channels, n)
        self._chan_scroll.setRange(0, max(0, n - visible))
        self._chan_scroll.setValue(self._channel_offset)

    def _filter_pad_samples(self) -> int:
        """Extra samples to read either side so a filter has room to work.

        A filter needs a stretch of signal several times longer than the
        period it is trying to resolve. MNE sizes its FIR kernel at roughly
        ``3.3 / cutoff`` seconds, so a 0.01 Hz high-pass wants 330 seconds,
        and handing it the ten seconds that happen to be on screen produces
        ``filter_length is longer than the signal, distortion is likely``
        followed by a trace that is mostly ringing.

        Reported from use on a real 1,449-second trigger channel, where the
        viewer filtered the visible window and drew the artefact.

        So the segment is PADDED: read wider, filter the wide stretch, then
        return only the part that is on screen. The padding is capped, at
        both the filter's own requirement and a ceiling, because the point
        of drawing a window at a time is not to read a forty-minute
        recording on every scroll.
        """
        lowest = None
        if self._current_filter and self._current_filter[0]:
            lowest = float(self._current_filter[0])
        if lowest is None or lowest <= 0:
            # A low-pass or a notch needs far less room; a second either
            # side settles any of them at the rates we see.
            if self._current_filter or self._notch_freq:
                return int(self._sfreq)
            return 0

        needed = int(round((3.3 / lowest) * self._sfreq))

        # The budget is in SAMPLES READ, not in seconds, because that is
        # what costs. One physio channel can afford the 330 seconds either
        # side that a 0.01 Hz high-pass wants; three hundred MEG channels
        # cannot, and would turn every scroll into a multi-second wait.
        channels = max(1, len(self._display_indices) or len(self._ch_names))
        affordable = int(_FILTER_SAMPLE_BUDGET / channels)
        if needed <= affordable:
            return needed

        # Not affordable. Say so, once per filter change rather than on
        # every redraw, because a trace that is quietly mostly ringing is
        # the worst of the three outcomes.
        if self._filter_pad_warned != (lowest, channels):
            self._filter_pad_warned = (lowest, channels)
            resolvable = 3.3 * self._sfreq / max(1, affordable)
            self.status_message.emit(
                f"A {lowest:g} Hz high-pass needs {needed / self._sfreq:.0f} s "
                f"of signal either side and {channels} channels only allow "
                f"{affordable / self._sfreq:.0f} s. Showing the closest this "
                f"view can resolve (about {resolvable:.3g} Hz); pick fewer "
                f"channels for the full depth."
            )
        return affordable

    def _get_data_segment(self, ch_indices, tmin, tmax):
        if self._raw is None:
            return None, None
        smin = max(0, int(tmin * self._sfreq))
        smax = min(self._n_samples, int(tmax * self._sfreq))
        if smax <= smin:
            return None, None

        pad = self._filter_pad_samples()
        pmin = max(0, smin - pad)
        pmax = min(self._n_samples, smax + pad)
        try:
            data, times = self._raw[ch_indices, pmin:pmax]
        except Exception:
            return None, None

        if self._current_filter or self._notch_freq:
            import mne

            l_freq = (
                self._current_filter[0]
                if self._current_filter and self._current_filter[0]
                else None
            )
            h_freq = (
                self._current_filter[1]
                if self._current_filter and self._current_filter[1]
                else None
            )
            if l_freq or h_freq:
                # A cut-off the AVAILABLE SIGNAL cannot resolve is refused,
                # not attempted. Padding buys room inside a long recording;
                # it cannot make a 2.2-second one longer, and MNE's answer
                # in that case is a warning followed by a trace that is
                # mostly ringing. Refusing and saying why is the only one of
                # the three outcomes a reader can act on.
                needed = (
                    int(round((3.3 / l_freq) * self._sfreq)) if l_freq else 0
                )
                if needed and data.shape[1] < needed:
                    if self._filter_short_warned != (l_freq, data.shape[1]):
                        self._filter_short_warned = (l_freq, data.shape[1])
                        self.status_message.emit(
                            f"A {l_freq:g} Hz high-pass needs "
                            f"{needed / self._sfreq:.0f} s of signal and this "
                            f"recording has {self._duration:.1f} s. The "
                            f"high-pass was not applied."
                        )
                    l_freq = None
                if l_freq or h_freq:
                    try:
                        data = mne.filter.filter_data(
                            data, self._sfreq, l_freq, h_freq, verbose=False,
                        )
                    except Exception:
                        pass
            if self._notch_freq and self._notch_freq > 0:
                try:
                    data = mne.filter.notch_filter(
                        data, self._sfreq, self._notch_freq, verbose=False,
                    )
                except Exception:
                    pass

        # Give back only what was asked for. The padding existed so the
        # filter had room, not so the user would see a wider window than
        # the one they set.
        if pad:
            lo = smin - pmin
            hi = lo + (smax - smin)
            data = data[:, lo:hi]
            times = times[lo:hi]

        # The gaps go back in LAST, after any filtering, so the filter saw a
        # continuous signal and the picture shows the truth. A gap in a
        # recording is a fact about the recording, and drawing it as a value
        # invents a sample that was never taken.
        if self._gaps is not None:
            try:
                mask = self._gaps[np.asarray(ch_indices), smin:smax]
                if mask.shape == data.shape:
                    data = np.where(mask, np.nan, data)
            except Exception:  # noqa: BLE001 - a mask mismatch must not blank the view
                pass
        return data, times

    def _redraw(self) -> None:
        self._plot_widget.clear()
        self._clear_labels()
        self._overlay_items = []
        self._shown_ch_info = []
        if self._raw is None:
            return
        n_disp = len(self._display_indices)
        if n_disp == 0:
            return
        vis = min(self._visible_channels, n_disp)
        start = min(self._channel_offset, max(0, n_disp - vis))
        shown = self._display_indices[start:start + vis]
        n_shown = len(shown)
        if n_shown == 0:
            return
        tmin = self._time_start
        tmax = min(self._time_start + self._time_window, self._duration)
        data, times = self._get_data_segment(shown, tmin, tmax)
        if data is None:
            return
        scale = self._scale_factor
        plot_item = self._plot_widget.getPlotItem()
        # Channel labels sit on the themed strip, so they always use the
        # theme text colour (NOT the canvas colour) - fixes wash-out when
        # "Dark plot" is on in light theme.
        label_color = self.palette().color(QPalette.ColorRole.Text).name()

        if self._normalize:
            for i in range(n_shown):
                trace = data[i]
                # nan-aware: a gap must not drag the centre or the range,
                # and np.ptp over a NaN returns NaN, which blanks the trace.
                spread = _nan_ptp(trace)
                rng = spread if spread > 0 else 1.0
                offset = n_shown - 1 - i
                y = ((trace - _nan_mean(trace)) / rng) * scale + offset
                self._plot_one(plot_item, times, y, shown[i], offset, label_color,
                               n_shown)
        else:
            type_ranges: dict = {}
            for i in range(n_shown):
                ct = self._ch_types[shown[i]]
                type_ranges.setdefault(ct, []).append(_nan_ptp(data[i]))
            type_scale = {}
            for ct, ranges in type_ranges.items():
                valid = [r for r in ranges if r > 0]
                type_scale[ct] = float(np.median(valid)) if valid else 1.0
            for i in range(n_shown):
                ct = self._ch_types[shown[i]]
                ref = type_scale.get(ct, 1.0) or 1.0
                trace = data[i]
                offset = n_shown - 1 - i
                y = ((trace - _nan_mean(trace)) / ref) * scale + offset
                self._plot_one(plot_item, times, y, shown[i], offset, label_color,
                               n_shown)

        plot_item.setXRange(tmin, tmax, padding=0)
        plot_item.setYRange(-0.5, n_shown - 0.5, padding=0.02)
        self._position_labels()
        if self._show_events:
            self._draw_events(tmin, tmax, n_shown)
        self.lbl_time.setText(
            f"{tmin:.1f} - {tmax:.1f} / {self._duration:.1f} s"
        )
        if self._duration > 0:
            denom = max(0.01, self._duration - self._time_window)
            self.sld_time.blockSignals(True)
            self.sld_time.setValue(max(0, min(1000, int((tmin / denom) * 1000))))
            self.sld_time.blockSignals(False)

    def _plot_one(self, plot_item, times, y, ch_idx, offset, label_color,
                  n_shown) -> None:
        ch_name = self._ch_names[ch_idx]
        ch_type = self._ch_types[ch_idx]
        colour = self._line_color or type_color(ch_type)
        pen = self._pg.mkPen(color=colour, width=self._pen_width(n_shown))
        times, y = _peak_decimate(times, y, self._plot_widget.width())
        # ``connect="finite"`` breaks the curve at a NaN instead of drawing
        # a line across it. That is the whole point of carrying the mask.
        plot_item.plot(times, y, pen=pen, connect="finite")
        self._add_channel_label(ch_name, n_shown, label_color)
        self._shown_ch_info.append((offset, ch_name, ch_type))

    def _clear_labels(self) -> None:
        self._channel_labels = []
        while self._label_layout.count() > 0:
            item = self._label_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_channel_label(self, ch_name, n_shown, color) -> None:
        # Elided in the MIDDLE, with the whole name on hover. The strip is a
        # fixed 68 pixels, and a long physio column like ``external_trigger``
        # was cut to "external_tri", losing the end that says what it is.
        lbl = ElidedLabel(ch_name, mode=Qt.TextElideMode.ElideMiddle)
        lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        lbl.setStyleSheet(f"QLabel {{ color: {color}; font-size: 9px; }}")
        # Sized and placed by _position_labels, once every trace is drawn
        # and the view knows where it put them.
        self._channel_labels.append(lbl)

    def _label_y(self, data_y: float) -> int:
        """Where ``data_y`` in the plot falls, in the label strip's pixels.

        Mapped through the view rather than divided out of a height, because
        the two columns do not share a geometry: the plot widget contains an
        x-axis and margins that no trace enters, and the traces are spread
        over the view's range plus its padding.
        """
        view = self._plot_widget
        vb = view.getPlotItem().getViewBox()
        scene = vb.mapViewToScene(QPointF(0.0, float(data_y)))
        on_screen = view.viewport().mapToGlobal(view.mapFromScene(scene))
        return self._label_area.mapFromGlobal(on_screen).y()

    def _position_labels(self, *_args) -> None:
        """Put each channel's name level with its trace.

        The names used to be ``plot_widget.height() / n`` pixels each,
        stacked from the top of the WIDGET. The widget includes the x-axis
        and margins the traces never enter, so every name was slightly too
        tall and the error accumulated down the column: measured on a real
        MEG recording, the last of 8 names sat 46 pixels below its trace and
        the last of 20 sat 48 below, a band and a half, pointing at a trace
        that was not there. And nothing recomputed them on a resize.

        Now each band's edges are mapped from the data coordinates the
        traces are drawn at, and each height is the difference of two ROUNDED
        edges, so rounding cannot accumulate either. Called after every
        redraw and whenever the view is resized.
        """
        labels = getattr(self, "_channel_labels", [])
        if not labels:
            return
        n = len(labels)
        lay = self._label_layout
        try:
            # Trace i is drawn at offset n-1-i, so its band runs from
            # n-0.5-i down to n-1.5-i.
            edges = [self._label_y(n - 0.5 - i) for i in range(n + 1)]
        except Exception:  # noqa: BLE001 - geometry not available yet
            edges = []
        while lay.count() > 0:
            lay.takeAt(0)
        if len(edges) != n + 1 or edges[-1] <= edges[0]:
            # Not laid out yet (a view that has never been shown has no
            # geometry to map through). Divide evenly for now; the first
            # resize, which showing it is, puts them right.
            height = max(1, int(self._plot_widget.height() / n))
            for lbl in labels:
                lbl.setFixedHeight(height)
                lay.addWidget(lbl)
            return
        top = max(0, edges[0])
        lay.addSpacing(top)
        for lbl, upper, lower in zip(labels, [top, *edges[1:-1]], edges[1:]):
            lbl.setFixedHeight(max(1, lower - upper))
            lay.addWidget(lbl)
        lay.addStretch(1)

    def _draw_events(self, tmin, tmax, n_shown) -> None:
        events = self._active_events()
        if not events:
            return
        visible = [e for e in events if tmin <= e[0] <= tmax]
        if len(visible) > 500:
            step = len(visible) // 500
            visible = visible[::step]
        labels = sorted({lbl for _, lbl in visible})
        color_map = {lbl: series_color(i) for i, lbl in enumerate(labels)}
        override = self._event_color_override
        width = self._event_line_width
        for t, lbl in visible:
            color = (
                override.name() if override and override.isValid()
                else color_map.get(lbl, series_color(0))
            )
            pen = self._pg.mkPen(color=color, width=width)
            line = self._pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
            self._plot_widget.addItem(line)
            self._overlay_items.append(line)
            if lbl:
                txt = self._pg.TextItem(str(lbl), color=color, anchor=(0.5, 1.0))
                txt.setPos(t, n_shown - 0.3)
                font = txt.textItem.font()
                font.setPointSize(7)
                txt.setFont(font)
                self._plot_widget.addItem(txt)
                self._overlay_items.append(txt)

    # ------------------------------------------------------------- handlers
    def _on_ch_type_changed(self, text) -> None:
        idx = self.cmb_ch_type.currentIndex()
        user_data = self.cmb_ch_type.itemData(idx)
        self._active_ch_type = user_data if user_data else text
        self._channel_offset = 0
        self._update_display_indices()
        self._redraw()

    def _on_n_changed(self, val) -> None:
        self._visible_channels = val
        self._update_channel_scrollbar()
        self._redraw()

    def _on_scale_changed(self, val) -> None:
        self._scale_factor = val
        self._redraw()

    def _on_window_changed(self, val) -> None:
        self._time_window = val
        self._redraw()

    def _on_time_slider(self, val) -> None:
        if self._duration <= 0:
            return
        max_start = max(0.0, self._duration - self._time_window)
        self._time_start = (val / 1000.0) * max_start
        self._redraw()

    def _on_channel_scroll(self, val) -> None:
        self._channel_offset = val
        self._redraw()

    def _install_drag_to_scrub(self) -> None:
        """Make a horizontal drag move the window through the recording.

        The ViewBox still receives the events; only what they MEAN changes.
        One pixel of drag is one pixel of signal, so the trace follows the
        cursor exactly as it would if the camera were moving, and the data
        under it is fetched as the window travels.

        Redraws are PACED BY THEIR OWN COST. Qt delivers a mouse-move event
        per pixel of travel, and a redraw here is a read, a filter and a
        repaint of every visible channel: redrawing on each one turned a
        100 pixel drag into fifty full redraws, which is seconds of frozen
        window on a 300-channel recording. So the next redraw waits until at
        least as long as the last one took has passed, which self-tunes.
        One physio channel updates every event; a wall of MEG updates a few
        times per gesture and lands exactly where the cursor left it,
        because the release always redraws.
        """
        view = self._plot_widget.getPlotItem().getViewBox()
        self._drag_anchor: Optional[float] = None
        self._drag_cost = 0.0        # seconds the last drag redraw took
        self._drag_last = 0.0        # when it finished

        def mouse_drag(event, axis=None):
            if self._raw is None or self._duration <= 0:
                event.ignore()
                return
            event.accept()
            width = max(1, view.width())
            # Seconds per pixel at the CURRENT window, so the trace tracks
            # the cursor whatever the zoom.
            per_pixel = self._time_window / width
            if event.isStart():
                self._drag_anchor = self._time_start
                self._drag_origin = event.buttonDownPos().x()
                return
            if self._drag_anchor is None:
                return
            moved = event.pos().x() - self._drag_origin
            # Dragging LEFT moves forward in time, the way a piece of paper
            # moves under a finger.
            target = self._drag_anchor - moved * per_pixel
            max_start = max(0.0, self._duration - self._time_window)
            target = max(0.0, min(target, max_start))
            finish = event.isFinish()
            if not finish and abs(target - self._time_start) < per_pixel / 2:
                return
            now = time.perf_counter()
            # Twice the measured cost, because the redraw is only the half
            # that happens here: the paint lands later, on the event loop,
            # and costs about as much again. Never more often than 60 Hz,
            # which no display would show anyway.
            interval = max(2.0 * self._drag_cost, 1.0 / 60.0)
            if not finish and now - self._drag_last < interval:
                return
            self._time_start = target
            self._redraw()
            self._drag_cost = time.perf_counter() - now
            self._drag_last = time.perf_counter()
            if finish:
                self._drag_anchor = None

        view.mouseDragEvent = mouse_drag
        # Deliberately NO double-click override. Reset view is a button, and
        # it also clears the filters and the channel selection, which is not
        # what somebody double-clicking a trace is asking for.

    def _on_plot_wheel(self, event) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            step = -1 if delta > 0 else 1
            new = max(
                self._chan_scroll.minimum(),
                min(self._chan_scroll.maximum(), self._chan_scroll.value() + step),
            )
            self._chan_scroll.setValue(new)
        event.accept()

    def _navigate(self, direction) -> None:
        step = self._time_window * 0.8
        if direction == "start":
            self._time_start = 0.0
        elif direction == "prev":
            self._time_start = max(0.0, self._time_start - step)
        elif direction == "next":
            self._time_start = min(
                max(0.0, self._duration - self._time_window),
                self._time_start + step,
            )
        elif direction == "end":
            self._time_start = max(0.0, self._duration - self._time_window)
        self._time_start = max(0.0, self._time_start)
        self._redraw()

    def _on_norm_toggled(self, checked) -> None:
        self._normalize = checked
        if self._raw is not None:
            self._redraw()

    def _on_dark_toggled(self, checked) -> None:
        self._dark_plot = checked
        self._apply_plot_theme()
        if self._raw is not None:
            self._redraw()

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            return
        if not self._shown_ch_info:
            return
        view_pt = self._plot_widget.getPlotItem().vb.mapSceneToView(pos)
        y = view_pt.y()
        nearest = ""
        min_dist = float("inf")
        for y_off, ch_name, _ in self._shown_ch_info:
            d = abs(y - y_off)
            if d < min_dist:
                min_dist = d
                nearest = ch_name
        if min_dist < 0.6:
            QToolTip.showText(QCursor.pos(), nearest, self._plot_widget)
        else:
            QToolTip.hideText()

    def _on_events_toggled(self, checked) -> None:
        self._show_events = checked
        if checked and self._raw is not None:
            self._jump_to_first_event_if_needed()
        if self._raw is not None:
            self._redraw()

    def _on_event_src_changed(self, text) -> None:
        self._event_source = text
        if self._show_events:
            self._jump_to_first_event_if_needed()
            self._redraw()

    def _jump_to_first_event_if_needed(self) -> None:
        """If events are enabled but none fall in the current window, scroll
        to the first one. Events often start well into a recording (the MEG
        sample's first trigger is at ~102 s), so without this "Show events"
        looks like it does nothing at t=0.
        """
        events = self._active_events()
        if not events:
            return
        tmin = self._time_start
        tmax = min(self._time_start + self._time_window, self._duration)
        if any(tmin <= t <= tmax for t, _ in events):
            return
        first = min(t for t, _ in events)
        max_start = max(0.0, self._duration - self._time_window)
        self._time_start = max(0.0, min(first - self._time_window * 0.1, max_start))
        self.status_message.emit(
            f"{len(events)} events; jumped to the first at {first:.1f}s"
        )

    def _on_event_width_changed(self, val) -> None:
        self._event_line_width = val
        if self._show_events:
            self._redraw()

    def _pick_event_color(self) -> None:
        color = QColorDialog.getColor(
            self._event_color_override or QColor("#ffcc00"),
            self,
            "Pick event line colour",
        )
        if color.isValid():
            self._event_color_override = color
            if self._show_events:
                self._redraw()

    def _reset_event_color(self) -> None:
        self._event_color_override = None
        if self._show_events:
            self._redraw()

    # --------------------------------------------------------------- filters
    def _apply_filters(self) -> None:
        self._filter_pad_warned = None
        self._filter_short_warned = None
        hp = self.spn_hp.value()
        lp = self.spn_lp.value()
        notch = self.spn_notch.value()
        self._current_filter = (hp if hp > 0 else None, lp if lp > 0 else None)
        self._notch_freq = notch if notch > 0 else None
        self._redraw()
        self.status_message.emit(
            f"Filter applied: HP={hp:g}Hz LP={lp:g}Hz Notch={notch:g}Hz"
        )

    def _reset_filters(self) -> None:
        self._filter_pad_warned = None
        self._filter_short_warned = None
        for w in (self.spn_hp, self.spn_lp, self.spn_notch):
            w.blockSignals(True)
            w.setValue(0.0)
            w.blockSignals(False)
        self._current_filter = None
        self._notch_freq = None
        self._redraw()
        self.status_message.emit("Filters reset")

    def _apply_resample(self) -> None:
        freq = self.spn_resample.value()
        if freq <= 0 or self._raw is None:
            return
        from ...workers import RecordingResampleWorker

        self.loading_changed.emit(True, f"Resampling to {freq:.0f} Hz…")
        self.btn_resample.setEnabled(False)
        worker = RecordingResampleWorker(self._raw, freq, parent=self)
        worker.finished_with_raw.connect(self._on_resampled)
        worker.failed.connect(self._on_resample_failed)
        worker.finished.connect(worker.deleteLater)
        self._resample_worker = worker
        worker.start()

    def _on_resampled(self, raw, freq) -> None:
        self._resample_worker = None
        self.loading_changed.emit(False, "")
        self.btn_resample.setEnabled(True)
        self._raw = raw
        self._sfreq = float(raw.info["sfreq"])
        self._n_samples = int(raw.n_times)
        self._duration = float(raw.times[-1]) if raw.n_times else 0.0
        self.spn_window.setMaximum(max(0.1, self._duration))
        self._redraw()
        self.status_message.emit(
            f"Resampled to {freq:.0f} Hz ({self._n_samples} samples)"
        )

    def _on_resample_failed(self, msg) -> None:
        self._resample_worker = None
        self.loading_changed.emit(False, "")
        self.btn_resample.setEnabled(True)
        QMessageBox.warning(self, "Resample error", msg)

    # ------------------------------------------------------------------ PSD
    def _show_psd(self) -> None:
        if self._raw is None:
            return
        from ...workers import RecordingComputeWorker

        self.loading_changed.emit(True, "Computing PSD…")
        self.btn_psd.setEnabled(False)
        raw = self._raw
        sfreq = self._sfreq

        # The channels the user is LOOKING AT, and the spectrum computed
        # from their SAMPLES rather than through ``Raw.compute_psd``.
        #
        # ``compute_psd`` refuses a stim channel outright, whatever picks it
        # is given: it treats picks as "data_or_ica" and a physio recording
        # of nothing but a trigger has none of those, so the spectrum came
        # back as "picks yielded no channels" on a real file. A spectrum is
        # a question about numbers, not about what kind of sensor produced
        # them, so MNE's channel-kind policy has no business deciding
        # whether it can be answered.
        #
        # Welch, because a viewer wants a stable average over the recording
        # rather than one long periodogram.
        picks = list(self._display_indices) or list(range(len(self._ch_names)))
        names = [self._ch_names[i] for i in picks]
        types = [self._ch_types[i] for i in picks]

        def _compute():
            from scipy.signal import welch

            data = raw.get_data(picks=picks)
            # Four-second windows where the recording allows: respiration
            # lives near 0.3 Hz and a shorter window puts the one feature a
            # respiratory trace is read for in the first bin.
            nperseg = int(min(data.shape[1], max(256, round(sfreq * 4))))
            freqs, power = welch(
                np.nan_to_num(np.asarray(data, dtype=np.float64)),
                fs=sfreq, nperseg=nperseg, axis=-1,
            )
            fmax = min(sfreq / 2.0, 150.0)
            keep = freqs <= fmax
            return {
                "freqs": freqs[keep],
                "data": np.atleast_2d(power)[:, keep],
                "ch_names": names,
                "ch_types": types,
            }

        worker = RecordingComputeWorker(_compute, parent=self)
        worker.finished_with_result.connect(self._on_psd_ready)
        worker.failed.connect(self._on_psd_failed)
        worker.finished.connect(worker.deleteLater)
        self._psd_worker = worker
        worker.start()

    def _on_psd_ready(self, result) -> None:
        self._psd_worker = None
        self.loading_changed.emit(False, "")
        self.btn_psd.setEnabled(True)
        dlg = PsdDialog(result, parent=self)
        dlg.show()
        self.status_message.emit("PSD computed")

    def _on_psd_failed(self, msg) -> None:
        self._psd_worker = None
        self.loading_changed.emit(False, "")
        self.btn_psd.setEnabled(True)
        QMessageBox.warning(self, "PSD error", msg)

    # ----------------------------------------------------------- selection
    def _open_channel_selector(self) -> None:
        if not self._ch_names:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Select channels")
        dlg.resize(360, 520)
        lay = QVBoxLayout(dlg)

        search = QLineEdit()
        search.setPlaceholderText("Search channels…")
        lay.addWidget(search)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        cmb = QComboBox()
        display = []
        for ct in self._available_ch_types:
            display.append("mag (axial grad)" if self._is_ctf and ct == "mag" else ct)
        cmb.addItems(["all"] + display)
        type_row.addWidget(cmb, 1)
        lay.addLayout(type_row)

        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for i, name in enumerate(self._ch_names):
            ct = self._ch_types[i]
            disp = "mag (axial grad)" if self._is_ctf and ct == "mag" else ct
            item = QListWidgetItem(f"{name}  [{disp}]")
            item.setData(Qt.ItemDataRole.UserRole, name)
            if self._selected_channels is None or name in self._selected_channels:
                item.setSelected(True)
            lst.addItem(item)
        lay.addWidget(lst, 1)

        qrow = QHBoxLayout()
        b_all = QPushButton("Select all")
        b_none = QPushButton("Select none")
        b_all.clicked.connect(lst.selectAll)
        b_none.clicked.connect(lst.clearSelection)
        qrow.addWidget(b_all)
        qrow.addWidget(b_none)
        lay.addLayout(qrow)

        def _filter():
            text = search.text().lower()
            ct = cmb.currentText()
            if ct == "mag (axial grad)":
                ct = "mag"
            for idx in range(lst.count()):
                item = lst.item(idx)
                name = item.data(Qt.ItemDataRole.UserRole)
                ch_type = self._ch_types[self._ch_names.index(name)]
                visible = True
                if text and text not in name.lower():
                    visible = False
                if ct != "all" and ch_type != ct:
                    visible = False
                item.setHidden(not visible)

        search.textChanged.connect(_filter)
        cmb.currentTextChanged.connect(lambda _=None: _filter())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected = {
                lst.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(lst.count())
                if lst.item(i).isSelected()
            }
            if len(selected) == len(self._ch_names):
                self._selected_channels = None
            else:
                self._selected_channels = selected
            self._channel_offset = 0
            self._update_display_indices()
            self._redraw()
            n = len(selected) if self._selected_channels else len(self._ch_names)
            self.status_message.emit(f"Displaying {n} channels")

    # ----------------------------------------------------------------- reset
    def enable_fit_all(self, enabled: bool = True) -> None:
        """Offer the Fit all button, for a recording that can survive it.

        Physio is a channel or four at 50 to 1000 Hz and fits in a window
        whole. MEG and EEG do not: the caller is the one that knows which it
        has, so the caller decides rather than this view guessing from a
        sample count that a filter could change.
        """
        self.btn_fit.setVisible(bool(enabled))

    def _drawn_count(self) -> int:
        """How many traces reach the pane.

        NOT the candidate pool: ``_display_indices`` holds every channel of
        the selected type and only the first ``_visible_channels`` of them
        are drawn.
        """
        return max(1, min(
            self._visible_channels or 1, len(self._display_indices) or 1,
        ))

    def max_pen_width(self, n_shown: Optional[int] = None) -> int:
        """The widest pen this view can draw at without going sluggish.

        A CAP, not a suggestion. Thickness is worth having and it is not
        free: above two traces a wider pen costs seven times the paint (see
        ``_THICK_LINE_MAX_TRACES``), and a drag pays that on every frame. A
        stored preference of six pixels must not turn a twenty-channel window
        into a slideshow just because it was set on a physio trace, so the
        cap applies to a chosen width as well as to the automatic one.
        """
        if n_shown is None:
            n_shown = self._drawn_count()
        return _THICK_LINE_WIDTH if n_shown <= _THICK_LINE_MAX_TRACES else 1

    def _pen_width(self, n_shown: Optional[int] = None) -> int:
        """The width to draw at: what was asked for, capped by what it costs."""
        if self.max_pen_width(n_shown) <= 1:
            return 1
        return self._line_width if self._line_width > 0 else _THICK_LINE_WIDTH

    def _open_line_style(self) -> None:
        """The Line popup. Live, so the plot behind it updates as you drag."""
        from .line_style_dialog import MAX_WIDTH, LineStyleDialog

        # Only the widths THIS view can draw at, so a number that would be
        # capped on the way out is never offered in the first place.
        ceiling = MAX_WIDTH if self.max_pen_width() > 1 else 1
        dlg = LineStyleDialog(
            self._line_width, self._line_color,
            allow_by_type=len(set(self._ch_types)) > 1,
            # Only the types THIS recording has: a MEG file offers mag, grad
            # and ref_meg, a physio run offers cardiac and trigger, and
            # neither is asked about a type it cannot draw.
            channel_types=self._present_channel_types(),
            max_width=ceiling,
            traces_shown=self._drawn_count(),
            parent=self,
        )
        dlg.changed.connect(self._set_line_style)
        dlg.type_colors_changed.connect(self._set_type_colors)
        dlg.exec()
        self._remember_line_style()

    def _present_channel_types(self) -> list[str]:
        """The channel types in this recording, in the order it lists them."""
        return list(dict.fromkeys(self._ch_types))

    def _set_line_style(self, width: int, colour: str) -> None:
        # Zero is kept: it is the automatic width, not a bad one.
        self._line_width = max(0, int(width))
        self._line_color = colour or None
        self._redraw()

    def _set_type_colors(self, mapping: dict) -> None:
        """Install and persist per-type colours, then redraw.

        Stored globally per type rather than per viewer, so a recording's
        traces and its spectrum agree: the PSD dialog resolves a kind's
        colour through the same function.
        """
        set_type_colors(mapping)
        try:
            from ..app_settings import AppSettings

            AppSettings.remember_type_colors(mapping)
        except Exception:  # noqa: BLE001 - a preference is not worth a crash
            log.debug("could not store the channel-type colours")
        self._redraw()

    def _remember_line_style(self) -> None:
        try:
            from ..app_settings import AppSettings

            AppSettings.remember_trace_style(
                self._line_width, self._line_color or "",
            )
        except Exception:  # noqa: BLE001 - a preference is not worth a crash
            log.debug("could not store the trace style")

    def _restore_line_style(self) -> None:
        """Zero is kept as zero: it is the automatic setting, not a bad one."""
        try:
            from ..app_settings import AppSettings

            settings = AppSettings.load()
            stored = int(settings.trace_line_width)
            self._line_width = stored if stored > 0 else _LINE_WIDTH_AUTO
            self._line_color = settings.trace_line_color or None
        except Exception:  # noqa: BLE001
            log.debug("could not read the trace style")

    def _fit_all(self) -> None:
        """Show the whole recording at once.

        The window length is a spin box, and winding it from ten seconds to
        forty minutes one step at a time is not a thing to ask of anyone.
        """
        if self._raw is None or self._duration <= 0:
            return
        self._time_start = 0.0
        self._time_window = self._duration
        self.spn_window.blockSignals(True)
        self.spn_window.setMaximum(max(0.1, self._duration))
        self.spn_window.setValue(self._duration)
        self.spn_window.blockSignals(False)
        self._redraw()

    def _sync_single_channel_controls(self) -> None:
        """Hide what a one-channel recording has no use for.

        A type filter, a channel picker and a "how many at once" count are
        three controls that can only ever say the same thing when there is
        one channel, and a control that cannot change anything is noise in
        a toolbar that is already full.
        """
        multi = len(self._ch_names) > 1
        for widget in (self._lbl_type, self.cmb_ch_type, self.btn_select,
                       self._lbl_count, self.spn_n, self._chan_scroll):
            widget.setVisible(multi)

    def _reset_view(self) -> None:
        if self._raw is None:
            return
        self._time_start = 0.0
        self._channel_offset = 0
        self._scale_factor = 1.0
        self._time_window = min(_DEFAULT_WINDOW_S, max(0.1, self._duration))
        self._active_ch_type = "all"
        self._current_filter = None
        self._notch_freq = None
        self._selected_channels = None
        self._visible_channels = _DEFAULT_VISIBLE
        self._normalize = False
        for w in (self.spn_scale, self.spn_window, self.spn_n, self.cmb_ch_type,
                  self.spn_hp, self.spn_lp, self.spn_notch, self.chk_norm):
            w.blockSignals(True)
        self.spn_scale.setValue(1.0)
        self.spn_window.setValue(self._time_window)
        self.spn_n.setValue(_DEFAULT_VISIBLE)
        self.cmb_ch_type.setCurrentText("all")
        self.spn_hp.setValue(0.0)
        self.spn_lp.setValue(0.0)
        self.spn_notch.setValue(0.0)
        self.chk_norm.setChecked(False)
        for w in (self.spn_scale, self.spn_window, self.spn_n, self.cmb_ch_type,
                  self.spn_hp, self.spn_lp, self.spn_notch, self.chk_norm):
            w.blockSignals(False)
        self._update_display_indices()
        self._update_channel_scrollbar()
        self._redraw()
        self.status_message.emit("View reset")

    # ---------------------------------------------------------------- unload
    def unload(self) -> None:
        for w in (self._resample_worker, self._psd_worker):
            if w is not None:
                w.cancel()
        self._resample_worker = None
        self._psd_worker = None
        self._plot_widget.clear()
        self._clear_labels()
        self._raw = None
        self._ch_names = []
        self._ch_types = []
        self._available_ch_types = []
        self._display_indices = []
        self._shown_ch_info = []
        self._overlay_items = []
        self._current_filter = None
        self._notch_freq = None
        self._selected_channels = None
        self._is_ctf = False
        self._duration = 0.0
        self._n_samples = 0
        self._tsv_events = []
        self._stim_events = []
        self.chk_events.setChecked(False)
        self.chk_events.setEnabled(False)
        self.lbl_time.setText("0.0 / 0.0 s")

    # ----------------------------------------------------------------- theme
    def _app_is_dark(self) -> bool:
        return self.palette().color(QPalette.ColorRole.Base).lightness() < 128

    def _apply_plot_theme(self) -> None:
        pg = self._pg
        dark_theme = self._app_is_dark()
        # In dark theme the canvas is already dark, so the "Dark plot" override
        # is redundant - disable + force it off.
        self.chk_dark.blockSignals(True)
        if dark_theme:
            self._dark_plot = False
            self.chk_dark.setChecked(False)
            self.chk_dark.setEnabled(False)
            self.chk_dark.setToolTip("Dark plot is automatic in dark theme")
        else:
            self.chk_dark.setEnabled(True)
            self.chk_dark.setToolTip(
                "Force a dark plot canvas (the canvas otherwise follows the "
                "app theme)."
            )
        self.chk_dark.blockSignals(False)

        if self._dark_plot and not dark_theme:
            bg = QColor("#11161d")
            x_fg = QColor("#cccccc")
        else:
            bg = self.palette().color(QPalette.ColorRole.Base)
            x_fg = self.palette().color(QPalette.ColorRole.Text)
        y_fg = self.palette().color(QPalette.ColorRole.Text)
        self._plot_widget.setBackground(bg)
        x_axis = self._plot_widget.getPlotItem().getAxis("bottom")
        x_axis.setPen(pg.mkPen(color=x_fg))
        x_axis.setTextPen(pg.mkPen(color=x_fg))
        y_axis = self._plot_widget.getPlotItem().getAxis("left")
        y_axis.setPen(pg.mkPen(color=y_fg))
        y_axis.setTextPen(pg.mkPen(color=y_fg))

    def repaint_for_palette(self, pal: dict) -> None:
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        for cmb in self.findChildren(QComboBox):
            view = cmb.view()
            targets = [cmb, view] + (
                [view.viewport()] if view and view.viewport() else []
            )
            for t in targets:
                style.unpolish(t)
                style.polish(t)
                t.update()
        for name, btn in (
            ("channels", self.btn_select),
            ("reset_view", self.btn_reset),
            ("close_data", self.btn_close),
            ("filter", self.btn_filter),
            ("resample", self.btn_resample),
            ("psd", self.btn_psd),
        ):
            icons.apply_button(btn, name)
        self._apply_plot_theme()
        if self._raw is not None:
            self._redraw()


__all__ = ["TimeSeriesView"]
