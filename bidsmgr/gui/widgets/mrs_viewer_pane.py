"""MR spectroscopy viewer (Editor center pane, ``mrs/`` kind).

Sister widget to :class:`NiftiViewerPane` and :class:`TimeSeriesView`. An
MRS file is a NIfTI by container only: the data block is a complex free
induction decay, and showing it as slices would show nothing at all. So
``mrs/`` routes here instead.

What it shows, and why each control exists
------------------------------------------
**A spectrum, in ppm, with the axis reversed.** That is the convention every
textbook and every fitting package uses, and a spectrum drawn the other way
round is a plausible-looking picture in which every metabolite is on the
wrong side of the water.

**Labelled metabolite positions.** NAA at 2.01, creatine at 3.03, choline at
3.22, myo-inositol at 3.56 and the rest, drawn as reference lines with
names. This is the difference between a row of unlabelled bumps and a
spectrum somebody can read: the question a reader brings is "is the NAA
where it should be", and with the lines on, that is answered by looking.
Verified on this lab's own data, where the strongest in-band peak lands at
3.02 ppm against creatine's textbook 3.03.

**Line broadening and zero-order phase**, because a raw transform is hard to
read and both are what every MRS package does before showing anyone
anything. Neither invents signal: broadening trades resolution for
signal-to-noise, and phase rotates a complex number.

**The FID itself**, as a second page, because when a spectrum looks wrong
the question is usually whether the time-domain signal decayed at all.

**The dynamics**, averaged or one at a time, because that is how a corrupted
repeat is found.

**A crosshair that reads out ppm and intensity**, because the question a
reader brings to a peak is "what shift is that", and answering it by
eyeballing the axis is how a shoulder gets mistaken for a metabolite.

**Line width and colour**, through the same ``LineStyleDialog`` the
MEG/EEG and physio views use, so the three cannot drift into three
different versions of the same two controls.

Threading and theme follow the house rules: the read and the transform run
on a ``QThread`` (never a pool, see CLAUDE.md guard 8b: this ends in an
FFT), and the pyqtgraph plot is handed the palette explicitly because it
reads no QSS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import CUR
from .flow_layout import flow
from .primitives import ElidedLabel, PaneHeader
from .spinner import BusySpinner

log = logging.getLogger(__name__)

#: Decimals on the numeric fields. Matches the time-series viewer, and for
#: the same reason: a 0.01 ppm shift is a real one.
_DECIMALS = 3

#: Where the metabolite names sit, as a fraction of the view height, one
#: entry per staggering row. Four rows is enough for a 1H table: crowded
#: enough to need it, not so many that the names walk into the peaks.
_LABEL_ROWS = (0.96, 0.89, 0.82, 0.75)


class MrsViewerPane(QWidget):
    """Center pane for a NIfTI-MRS file."""

    status_message = pyqtSignal(str)
    loading_changed = pyqtSignal(bool, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")
        self.setMinimumWidth(0)

        self._path: Optional[Path] = None
        self._root: Optional[Path] = None
        self._data: Optional[dict] = None
        self._worker = None
        self._pg = None
        self._plot = None
        self._curve = None
        self._markers: list = []
        self._marker_shifts: list[float] = []
        self._cross = None
        self._ppm_range: Optional[tuple[float, float]] = None
        self._line_width = 2
        self._line_color: Optional[str] = None
        self._restore_line_style()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = PaneHeader("Spectroscopy")
        outer.addWidget(self._header)

        self._bar = QFrame()
        self._bar.setObjectName("toolbar")
        self._controls = flow(self._bar, h_spacing=10, v_spacing=4)
        self._controls.setContentsMargins(10, 4, 10, 4)
        outer.addWidget(self._bar)
        self._build_controls()

        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        holder = QWidget()
        holder.setLayout(self._stack)
        outer.addWidget(holder, 1)

        self._hint = QLabel("Select a spectroscopy file.")
        self._hint.setObjectName("pane-hint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        self._stack.addWidget(self._hint)

        self._spinner_page = QWidget()
        sp = QVBoxLayout(self._spinner_page)
        sp.addStretch(1)
        self._spinner = BusySpinner()
        sp.addWidget(self._spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        self._loading_label = ElidedLabel(
            "", mode=Qt.TextElideMode.ElideMiddle,
        )
        self._loading_label.setObjectName("pane-hint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sp.addWidget(self._loading_label)
        sp.addStretch(1)
        self._stack.addWidget(self._spinner_page)

        self._plot_page = QWidget()
        pv = QVBoxLayout(self._plot_page)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        self._plot_holder = QWidget()
        self._plot_holder_layout = QVBoxLayout(self._plot_holder)
        self._plot_holder_layout.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(self._plot_holder, 1)
        foot = QHBoxLayout()
        foot.setContentsMargins(10, 0, 10, 2)
        foot.setSpacing(10)
        self._caption = ElidedLabel("")
        self._caption.setObjectName("sidecar-footer-summary")
        foot.addWidget(self._caption, 1)
        self._readout = QLabel("")
        self._readout.setObjectName("sidecar-footer-summary")
        self._readout.setToolTip(
            "Where the cursor is. The question a reader brings to a peak is "
            "which shift it sits at, and reading that off the axis by eye is "
            "how a shoulder gets mistaken for a metabolite."
        )
        foot.addWidget(self._readout, 0)
        pv.addLayout(foot)
        self._stack.addWidget(self._plot_page)

        self._stack.setCurrentIndex(0)
        self._bar.setVisible(False)

    # -- controls ----------------------------------------------------------

    def _labelled(self, text: str, widget: QWidget, tip: str) -> None:
        """A control with its caption, added as ONE wrapping item."""
        holder = QHBoxLayout()
        holder.setContentsMargins(0, 0, 0, 0)
        holder.setSpacing(4)
        label = QLabel(text)
        label.setToolTip(tip)
        widget.setToolTip(tip)
        holder.addWidget(label)
        holder.addWidget(widget)
        self._controls.addLayout(holder)

    def _build_controls(self) -> None:
        self._domain = QComboBox()
        self._domain.addItems(["Spectrum", "FID (time domain)"])
        # Switching domain changes the x UNITS, so the span has to be
        # re-applied; the others (part, broadening, phase, repeat) leave a
        # zoom alone on purpose, because more broadening on the peak you are
        # looking at should not throw you back to the whole spectrum.
        self._domain.currentIndexChanged.connect(self._on_domain_changed)
        self._labelled(
            "Show", self._domain,
            "The spectrum is what a spectrum is read from. The FID is the "
            "signal the scanner actually measured, and is what to look at "
            "when a spectrum looks wrong: the question is usually whether "
            "the time-domain signal decayed at all.",
        )

        self._part = QComboBox()
        self._part.addItems(["Magnitude", "Real", "Imaginary", "Phase"])
        self._part.currentIndexChanged.connect(lambda _i: self._redraw())
        self._labelled(
            "Part", self._part,
            "A spectrum is complex. Magnitude needs no phasing and is the "
            "safe default; the real part is what quantification uses, once "
            "the phase is right.",
        )

        self._broadening = QDoubleSpinBox()
        self._broadening.setRange(0.0, 50.0)
        self._broadening.setDecimals(_DECIMALS)
        self._broadening.setSingleStep(0.5)
        self._broadening.setValue(2.0)
        self._broadening.setSuffix(" Hz")
        self._broadening.setKeyboardTracking(False)
        self._broadening.valueChanged.connect(lambda _v: self._redraw())
        self._labelled(
            "Line broadening", self._broadening,
            "Multiplies the FID by a decaying exponential before the "
            "transform, trading resolution for signal-to-noise. Every MRS "
            "package applies a few Hz, because the tail of an in-vivo FID "
            "is noise with no signal left in it.",
        )

        self._phase = QDoubleSpinBox()
        self._phase.setRange(-180.0, 180.0)
        self._phase.setDecimals(1)
        self._phase.setSingleStep(5.0)
        self._phase.setSuffix(" deg")
        self._phase.setKeyboardTracking(False)
        self._phase.valueChanged.connect(lambda _v: self._redraw())
        self._labelled(
            "Phase", self._phase,
            "Zero-order phase. An FID is rarely perfectly phased as "
            "acquired, and an unphased real part shows peaks dipping below "
            "the baseline, which reads as an artefact rather than a phase.",
        )

        self._dynamic = QSpinBox()
        self._dynamic.setRange(0, 0)
        self._dynamic.setSpecialValueText("Average")
        self._dynamic.setKeyboardTracking(False)
        self._dynamic.valueChanged.connect(lambda _v: self._redraw())
        self._labelled(
            "Repeat", self._dynamic,
            "The repeats exist to be averaged, and that is where the "
            "signal-to-noise comes from. Stepping through them one at a "
            "time is how a corrupted repeat gets found.",
        )

        self._show_metabolites = QCheckBox("Metabolites")
        self._show_metabolites.setChecked(True)
        self._show_metabolites.setToolTip(
            "Draw the reference positions of the metabolites a 1H brain "
            "spectrum is read for, with their names: NAA at 2.01 ppm, "
            "creatine at 3.03, choline at 3.22, myo-inositol at 3.56. "
            "Shown only for 1H, because 1H labels on a 31P spectrum would "
            "be in the wrong places."
        )
        self._show_metabolites.toggled.connect(lambda _c: self._redraw())
        self._controls.addWidget(self._show_metabolites)

        self._clip = QCheckBox("Standard window")
        self._clip.setChecked(True)
        self._clip.setToolTip(
            "Show 0.2 to 4.2 ppm, the window a 1H brain spectrum is "
            "conventionally displayed in. Outside it there is water on one "
            "side and lipid on the other, and neither is what the scan was "
            "run for."
        )
        self._clip.toggled.connect(self._on_clip_toggled)
        self._controls.addWidget(self._clip)

        self._line_btn = QPushButton("  Line")
        self._line_btn.setObjectName("tb-btn")
        self._line_btn.setToolTip(
            "Trace thickness and colour. The same popup the MEG/EEG and "
            "physio viewers use, so the three stay the same tool."
        )
        self._line_btn.clicked.connect(self._open_line_style)
        self._controls.addWidget(self._line_btn)

        self._fit_btn = QPushButton("  Fit all")
        self._fit_btn.setObjectName("tb-btn")
        self._fit_btn.setToolTip(
            "Show everything that was acquired, water and lipid included. "
            "The whole sampled band, not the window a brain spectrum is "
            "conventionally read in."
        )
        self._fit_btn.clicked.connect(self._fit_all)
        self._controls.addWidget(self._fit_btn)

        self._reset_btn = QPushButton("  Reset view")
        self._reset_btn.setObjectName("tb-btn")
        self._reset_btn.setToolTip(
            "Back to the conventional window, after zooming or dragging."
        )
        self._reset_btn.clicked.connect(self._reset_view)
        self._controls.addWidget(self._reset_btn)

        self._bar.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum,
        )

    # -- binding -----------------------------------------------------------

    def current_file(self) -> Optional[Path]:
        return self._path

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        """Bind a NIfTI-MRS file, or ``None`` to clear."""
        self._cancel()
        self._path = Path(path) if path else None
        self._root = Path(root) if root else None
        self._data = None
        if self._path is None:
            self._bar.setVisible(False)
            self._stack.setCurrentIndex(0)
            return

        self._header.setText(self._path.name)
        self._loading_label.setText(str(self._path.name))
        self._spinner.set_busy(True)
        self._stack.setCurrentIndex(1)
        self.loading_changed.emit(True, "Reading the spectrum...")
        self._start_read()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

    def _start_read(self) -> None:
        """Read on a QThread. Never a pool: this ends in an FFT."""
        from ...workers.meeg_recording_loader import RecordingComputeWorker
        from .mrs_spectrum import read_mrs

        path = self._path

        def work():
            return path, read_mrs(path)

        worker = RecordingComputeWorker(work, parent=self)
        worker.finished_with_result.connect(self._on_read)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_read(self, result) -> None:
        path, data = result
        self._worker = None
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        if path != self._path:
            return
        if data is None:
            self._bar.setVisible(False)
            self._hint.setText(
                f"{path.name} is in an mrs/ folder but carries no NIfTI-MRS "
                f"header, so there is no spectrum to show. The acquisition "
                f"parameters a spectrum needs (the dwell time, the "
                f"spectrometer frequency and the nucleus) live in that "
                f"header, and nothing can be derived without them."
            )
            self._stack.setCurrentIndex(0)
            return

        self._data = data
        n_dyn = int(data.get("n_dynamics", 1))
        self._dynamic.blockSignals(True)
        self._dynamic.setRange(0, max(0, n_dyn))
        self._dynamic.setValue(0)
        self._dynamic.blockSignals(False)
        self._dynamic.setEnabled(n_dyn > 1)

        nucleus = str(data.get("nucleus", "1H"))
        self._show_metabolites.setEnabled(nucleus.upper() == "1H")
        self._clip.setEnabled(nucleus.upper() == "1H")

        self._bar.setVisible(True)
        self._ensure_plot()
        self._stack.setCurrentIndex(2)
        self._redraw()
        self._reset_view()

    def _on_failed(self, message: str) -> None:
        self._worker = None
        self._spinner.set_busy(False)
        self.loading_changed.emit(False, "")
        self._bar.setVisible(False)
        self._hint.setText(f"Could not read the spectrum: {message}")
        self._stack.setCurrentIndex(0)

    # -- drawing -----------------------------------------------------------

    def _ensure_plot(self) -> None:
        if self._plot is not None:
            return
        import pyqtgraph as pg

        self._pg = pg
        self._plot = pg.PlotWidget()
        self._plot.setMinimumWidth(60)
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot_holder_layout.addWidget(self._plot)

        view = self._plot.getPlotItem().getViewBox()
        # Scale the height to WHAT IS ON SCREEN. Zoomed into the 2 to 4 ppm
        # region of an in-vivo spectrum, the water residual sits off screen
        # and a whole-trace y range flattens NAA, creatine and choline into
        # three bumps at the bottom of the pane. Autoscaling to the visible
        # slice is what makes zooming in show more rather than less.
        view.setAutoVisible(y=True)
        view.enableAutoRange(axis=view.YAxis)
        # Re-stagger as the window changes: the labels keep their size while
        # the gaps between them grow in pixels, so what fits on one row at
        # full width does not at a tenth of it, and the other way round.
        view.sigXRangeChanged.connect(self._restagger)
        self._wire_crosshair()
        self.repaint_for_palette(CUR())

    def _wire_crosshair(self) -> None:
        """A dashed cross following the cursor, with the value under it."""
        pg = self._pg
        palette = CUR()
        pen = pg.mkPen(
            palette.get("muted", "#8b949e"), width=1,
            style=Qt.PenStyle.DashLine,
        )
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        vline.setZValue(20)
        hline.setZValue(20)
        self._plot.addItem(vline, ignoreBounds=True)
        self._plot.addItem(hline, ignoreBounds=True)

        def moved(event):
            pos = event[0]
            if self._plot is None or not self._plot.sceneBoundingRect().contains(pos):
                return
            point = self._plot.getPlotItem().getViewBox().mapSceneToView(pos)
            unit = "s" if self._domain.currentIndex() == 1 else "ppm"
            digits = 4 if unit == "s" else 2
            self._readout.setText(
                f"{point.x():.{digits}f} {unit}   ·   {point.y():.3g}"
            )
            vline.setPos(point.x())
            hline.setPos(point.y())

        proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved, rateLimit=30, slot=moved,
        )
        # The proxy has to outlive this method or the signal stops arriving.
        self._cross = {"v": vline, "h": hline, "proxy": proxy}

    def _redraw(self) -> None:
        if self._data is None or self._plot is None:
            return
        from .mrs_spectrum import (
            combine,
            default_ppm_range,
            metabolites_for,
            part,
            spectrum,
        )

        pg = self._pg
        plot_item = self._plot.getPlotItem()
        plot_item.clear()
        self._markers = []
        self._marker_shifts = []
        self._readd_crosshair()

        data = self._data
        which = self._dynamic.value()
        fid = combine(
            data["fid"],
            mode="single" if which > 0 else "mean",
            index=which - 1,
        )
        palette = CUR()
        pen = pg.mkPen(
            self._line_color or palette.get("accent", "#58a6ff"),
            width=self._line_width,
        )
        component = self._part.currentText().lower()
        view = plot_item.getViewBox()

        if self._domain.currentIndex() == 1:
            # The FID: what the scanner measured, against real seconds.
            t = np.arange(fid.shape[0]) * float(data["dwell"])
            y = part(fid, component)
            plot_item.plot(t, y, pen=pen)
            view.invertX(False)
            self._limit_pan(float(t.min()), float(t.max()) if t.size else 1.0)
            self._plot.setLabel("bottom", "Time", units="s")
            self._plot.setLabel("left", component.capitalize())
            self._caption.setText(
                f"Free induction decay, {fid.shape[0]:,} points at "
                f"{float(data['dwell']) * 1e6:.0f} us, "
                f"{data['n_dynamics']} repeat(s)"
            )
            return

        ppm, _hz, spec = spectrum(
            fid, data["dwell"], data["spectrometer_mhz"], data["nucleus"],
            line_broadening_hz=self._broadening.value(),
            phase_deg=self._phase.value(),
        )
        y = part(spec, component)
        plot_item.plot(ppm, y, pen=pen)

        # Chemical shift runs RIGHT TO LEFT. Every textbook, every fitting
        # package, every paper. A spectrum drawn the other way is a
        # plausible picture with every metabolite on the wrong side.
        plot_item.getViewBox().invertX(True)
        self._plot.setLabel("bottom", "Chemical shift", units="ppm")
        self._plot.setLabel("left", component.capitalize())

        if self._show_metabolites.isChecked():
            self._draw_metabolites(metabolites_for(str(data["nucleus"])))

        lo, hi = default_ppm_range(str(data["nucleus"]), ppm)
        rng = (lo, hi) if self._clip.isChecked() else (float(ppm.min()), float(ppm.max()))
        self._ppm_range = rng
        self._limit_pan(float(ppm.min()), float(ppm.max()))

        nuc = str(data["nucleus"])
        self._caption.setText(
            f"{nuc} spectrum, {fid.shape[0]:,} points, "
            f"{1.0 / float(data['dwell']):.0f} Hz wide at "
            f"{float(data['spectrometer_mhz']):.2f} MHz, "
            f"{data['n_dynamics']} repeat(s)"
            + (f", {self._broadening.value():g} Hz broadening"
               if self._broadening.value() else "")
        )

    def _draw_metabolites(self, table) -> None:
        """Reference lines with names. The reason to open this viewer."""
        if not table:
            return
        pg = self._pg
        palette = CUR()
        colour = palette.get("muted", "#8b949e")
        for name, shift, description in table:
            line = pg.InfiniteLine(
                pos=shift, angle=90, movable=False,
                pen=pg.mkPen(colour, width=1, style=Qt.PenStyle.DashLine),
                label=name,
                labelOpts={
                    "position": _LABEL_ROWS[0],
                    "color": palette.get("text", "#c9d1d9"),
                    "movable": False,
                    "fill": None,
                },
            )
            line.setToolTip(f"{name}: {description}, {shift:g} ppm")
            self._plot.addItem(line, ignoreBounds=True)
            self._markers.append(line)
            self._marker_shifts.append(float(shift))
        self._restagger()

    def _restagger(self, *_args) -> None:
        """Move each metabolite name to a row where nothing covers it.

        Called on every range change as well as after drawing, because which
        names collide is a fact about the CURRENT window: creatine at 3.03
        and choline at 3.22 are a fifth of a ppm apart, which is a fifth of
        the pane at full width and the whole pane zoomed in.
        """
        if not self._markers or self._plot is None:
            return
        from .mrs_spectrum import stagger

        view = self._plot.getPlotItem().getViewBox()
        (x0, x1), _y = view.viewRange()
        rows = stagger(self._marker_shifts, abs(x1 - x0), rows=len(_LABEL_ROWS))
        for line, row in zip(self._markers, rows):
            label = getattr(line, "label", None)
            if label is not None:
                label.setPosition(_LABEL_ROWS[min(row, len(_LABEL_ROWS) - 1)])

    def _readd_crosshair(self) -> None:
        """Put the cross back after a clear, which takes every item with it."""
        if self._cross is None or self._plot is None:
            return
        for key in ("v", "h"):
            self._plot.addItem(self._cross[key], ignoreBounds=True)

    def _limit_pan(self, low: float, high: float) -> None:
        """Stop a drag from leaving the data behind.

        Without this, panning walks off the end of the spectrum into an empty
        pane with no way back but Reset, which is the complaint that produced
        this method.
        """
        if self._plot is None or not np.isfinite([low, high]).all():
            return
        margin = max(abs(high - low) * 0.02, 1e-9)
        self._plot.getPlotItem().getViewBox().setLimits(
            xMin=min(low, high) - margin, xMax=max(low, high) + margin,
        )

    # -- view --------------------------------------------------------------

    def _apply_x_range(self) -> None:
        """Put the view on the span the current settings ask for."""
        if self._plot is None:
            return
        plot_item = self._plot.getPlotItem()
        if self._domain.currentIndex() == 0 and self._ppm_range:
            plot_item.setXRange(*self._ppm_range, padding=0.02)
        else:
            plot_item.getViewBox().autoRange(padding=0.02)
        self._restagger()

    def _on_domain_changed(self, _index: int) -> None:
        self._redraw()
        self._apply_x_range()

    def _on_clip_toggled(self, _checked: bool) -> None:
        """Redrawing decides the span; something has to go and use it.

        Without the second call the checkbox recomputed the range and left
        the view where it was, so ticking it looked like it did nothing.
        """
        self._redraw()
        self._apply_x_range()

    def _reset_view(self) -> None:
        if self._plot is None:
            return
        view = self._plot.getPlotItem().getViewBox()
        view.enableAutoRange(axis=view.YAxis)
        self._apply_x_range()

    def _fit_all(self) -> None:
        """The whole sampled band, water and lipid included.

        Distinct from Reset view, which returns to whichever span the
        controls ask for. This is the one to reach for when a peak is
        suspected outside the conventional window, and the only way to see
        the water residual the sequence was meant to suppress.
        """
        if self._plot is None:
            return
        if self._domain.currentIndex() == 0 and self._clip.isChecked():
            # Unticking redraws AND applies, which is the whole band.
            self._clip.setChecked(False)
            return
        view = self._plot.getPlotItem().getViewBox()
        view.enableAutoRange(axis=view.YAxis)
        self._apply_x_range()

    def _open_line_style(self) -> None:
        """The Line popup, shared with the other viewers. Live, not on OK."""
        from .line_style_dialog import LineStyleDialog

        dlg = LineStyleDialog(
            self._line_width, self._line_color,
            # One trace, so there is neither a type to colour by nor a list
            # of types to configure.
            allow_by_type=False,
            channel_types=(),
            parent=self,
        )
        dlg.changed.connect(self._set_line_style)
        dlg.exec()
        self._remember_line_style()

    def _set_line_style(self, width: int, colour: str) -> None:
        # Automatic (zero) means two here: see _restore_line_style.
        self._line_width = int(width) if int(width) > 0 else 2
        self._line_color = colour or None
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
        """A stored width of zero means automatic, and here that is two.

        The shared setting has to serve a wall of MEG as well as this pane,
        so automatic asks how much is on screen. A spectrum is ONE trace of a
        few thousand points whatever the recording, so the answer never
        changes and the thicker line is always affordable.
        """
        try:
            from ..app_settings import AppSettings

            settings = AppSettings.load()
            stored = int(settings.trace_line_width)
            self._line_width = stored if stored > 0 else 2
            self._line_color = settings.trace_line_color or None
        except Exception:  # noqa: BLE001
            log.debug("could not read the trace style")

    # -- theme -------------------------------------------------------------

    def repaint_for_palette(self, pal: dict) -> None:
        """pyqtgraph reads no QSS, so the palette is handed down."""
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()
        if self._plot is None:
            return
        self._plot.setBackground(pal.get("bg", "#0d1117"))
        for axis in ("left", "bottom"):
            item = self._plot.getPlotItem().getAxis(axis)
            item.setPen(pal.get("border", "#30363d"))
            item.setTextPen(pal.get("muted", "#8b949e"))
        if self._cross is not None and self._pg is not None:
            pen = self._pg.mkPen(
                pal.get("muted", "#8b949e"), width=1,
                style=Qt.PenStyle.DashLine,
            )
            for key in ("v", "h"):
                self._cross[key].setPen(pen)
        self._redraw()


__all__ = ["MrsViewerPane"]
