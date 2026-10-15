"""The spectrum canvas: an MRS spectrum (or its FID), read the way a
spectroscopist reads one.

* the ppm axis runs RIGHT TO LEFT, as every textbook and fitting package
  draws it; drawn the other way, every metabolite is on the wrong side of the
  water and the picture still looks plausible;
* labelled metabolite positions (NAA 2.01, creatine 3.03, choline 3.22,
  myo-inositol 3.56, ...), staggered into rows so neighbours do not cover
  each other, re-staggered on every zoom because which names collide is a
  fact about the visible span;
* the height is FITTED to the visible ppm window on every pan and zoom,
  leaving the residual water band out (``Scene.spectrum.exclude_water``), so
  a water peak fifty times NAA runs off the top instead of flattening every
  metabolite; the wheel zooms ppm only, Shift or Alt with the wheel (or
  ``[`` / ``]``) scales the height to look under the tall peaks;
* panning stops at the data, so a drag cannot walk into an empty pane;
* a crosshair reading out ppm and intensity;
* the water reference overlaid, scaled to the spectrum (M6);
* interactive phasing: Ctrl+drag turns the zero-order phase, Ctrl+Shift+drag
  the first-order one (M3).

Everything is drawn from the scene (``Scene.spectrum``) and a
:class:`~bidsmgr.viz.data.spectrum.SpectrumSource`; every change is a command.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ....viz.commands import spectrum as spcmd
from ....viz.compute import mrs as M
from ..bridge import connect_while_alive
from ..context import ViewerContext

log = logging.getLogger(__name__)

#: Where metabolite names sit, as a fraction of the view height, per row.
LABEL_ROWS = (0.96, 0.89, 0.82, 0.75)
#: Degrees of zero-order phase per pixel of drag; ms of first-order per pixel.
_PHASE0_PER_PX = 0.5
_PHASE1_PER_PX = 0.002


class SpectrumCanvas(QWidget):
    def __init__(self, ctx: ViewerContext, parent=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setObjectName("pane-dark")
        import pyqtgraph as pg

        self._pg = pg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumWidth(60)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        pi = self.plot.getPlotItem()
        pi.setMenuEnabled(False)
        for name in ("left", "bottom"):
            pi.getAxis(name).enableAutoSIPrefix(False)
        lay.addWidget(self.plot)
        self._vb = pi.getViewBox()
        # The height is ours to fit (see _fit_y): pyqtgraph's auto-range
        # switches itself off at the first wheel or drag, and fits min/max of
        # everything visible, water included.
        self._vb.disableAutoRange(axis=self._vb.YAxis)
        self._vb.setMouseEnabled(x=True, y=False)
        self._vb.sigXRangeChanged.connect(self._restagger)
        self._vb.sigXRangeChanged.connect(lambda *_a: self._fit_y())
        self._wheel_default = self._vb.wheelEvent
        self._vb.wheelEvent = self._on_wheel
        self._xy: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._drag_default = self._vb.mouseDragEvent
        self._vb.mouseDragEvent = self._on_drag
        self.curve = pg.PlotCurveItem(antialias=True)
        self.ref_curve = pg.PlotCurveItem(antialias=True)
        pi.addItem(self.curve)
        # The reference is drawn, never fitted: its own scale must not set
        # the height of the spectrum it is compared with.
        pi.addItem(self.ref_curve, ignoreBounds=True)
        # Metabolite markers are POOLED: reused and hidden, never removed.
        # pyqtgraph takes an item out of the scene before detaching it from
        # its view, and Qt asking a line for its bounds in between was a
        # segfault waiting for the right timing. ``_markers`` holds the ones
        # on screen; ``_marker_pool`` every one ever made.
        self._marker_pool: list = []
        self._markers: list = []
        self._marker_shifts: list[float] = []
        self._ppm_range: Optional[tuple[float, float]] = None
        self.readout = ""
        self._phase_anchor = None
        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        for line in (self._vline, self._hline):
            line.setZValue(20)
            pi.addItem(line, ignoreBounds=True)
        self._proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30,
                                     slot=self._on_hover)
        ctx.qstore.changed.connect(self._on_changed)
        connect_while_alive(ctx.theme_hub.changed, self, lambda w, _t: w.redraw())
        connect_while_alive(ctx.settings_hub.changed, self, lambda w, _s: w.redraw())

    @property
    def source(self):
        return spcmd.source(self.ctx.store)

    def _on_changed(self, paths) -> None:
        if not self.isVisible():
            return
        if "spectrum" in paths or "scene" in paths or any(p.startswith("sources") for p in paths):
            sp = self.ctx.scene.spectrum
            key = (sp.domain, sp.standard_window)
            reframe = key != getattr(self, "_frame_key", None) or "scene" in paths
            self.redraw()
            if reframe:
                self._frame_key = key
                self.reset_view()
            else:
                self._fit_y()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.redraw()
        self.reset_view()

    # ------------------------------------------------------------------
    def _line_width(self) -> int:
        # A spectrum is ONE trace of a few thousand points, so the thicker
        # line is always affordable: automatic means two here.
        chosen = self.ctx.settings.traces.line_width
        return chosen if chosen > 0 else 2

    def _apply_theme(self) -> None:
        theme = self.ctx.theme
        pg = self._pg
        self.plot.setBackground(theme.plot_background)
        pi = self.plot.getPlotItem()
        for name in ("left", "bottom"):
            pi.getAxis(name).setPen(theme.grid)
        from .. import fonts

        fonts.style_axes(pi, theme.dim)
        pen = pg.mkPen(theme.dim, width=1, style=Qt.PenStyle.DashLine)
        self._vline.setPen(pen)
        self._hline.setPen(pen)

    def _titles(self, bottom: str, units: str, left: str) -> None:
        """The axes' titles at the app's font size."""
        from .. import fonts

        pi = self.plot.getPlotItem()
        colour = self.ctx.theme.dim
        fonts.axis_title(pi, "bottom", bottom, colour, units=units)
        fonts.axis_title(pi, "left", left, colour)

    def redraw(self) -> None:
        if not self.isVisible():
            return
        self._apply_theme()
        src = self.source
        pg = self._pg
        theme = self.ctx.theme
        sp = self.ctx.scene.spectrum
        self._show_metabolites(())
        if src is None:
            self.curve.setData([], [])
            self.ref_curve.setData([], [])
            self._xy = None
            return
        fid = src.select(dynamic=sp.repeat, edit=sp.edit)
        colour = self.ctx.settings.traces.line_color or theme.accent
        self.curve.setPen(pg.mkPen(colour, width=self._line_width()))
        if sp.domain == "fid":
            t = np.arange(fid.shape[0]) * src.dwell
            yt = M.part(fid, sp.part)
            self.curve.setData(t, yt)
            self._xy = (t, np.asarray(yt, dtype=float))
            self.ref_curve.setData([], [])
            self._vb.invertX(False)
            self._limit_pan(float(t.min()), float(t.max()) if t.size else 1.0)
            self._titles("Time", "s", self._y_label(sp.part))
            self._ppm_range = None
            return
        ppm, _hz, spec = M.spectrum(fid, src.dwell, src.spectrometer_mhz, src.nucleus,
                                    line_broadening_hz=sp.lb_hz, phase_deg=sp.phase0,
                                    phase1_ms=sp.phase1_ms)
        y = M.part(spec, sp.part)
        self.curve.setData(ppm, y)
        self._xy = (np.asarray(ppm, dtype=float), np.asarray(y, dtype=float))
        self._draw_reference(sp, ppm, y, src)
        self._vb.invertX(True)
        self._titles("Chemical shift", "ppm", self._y_label(sp.part))
        if sp.metabolites:
            self._show_metabolites(M.metabolites_for(src.nucleus))
        lo, hi = M.default_ppm_range(src.nucleus, ppm)
        self._ppm_range = ((lo, hi) if sp.standard_window
                           else (float(ppm.min()), float(ppm.max())))
        self._limit_pan(float(ppm.min()), float(ppm.max()))

    @staticmethod
    def _y_label(part: str) -> str:
        # Arbitrary units, said so: an MRS intensity is not a calibrated number.
        return "Phase (deg)" if part == "phase" else "Intensity (a.u.)"

    def _exclude(self):
        src = self.source
        sp = self.ctx.scene.spectrum
        if (sp.domain == "spectrum" and sp.exclude_water and src is not None
                and src.nucleus.upper() == "1H" and sp.part != "phase"):
            return M.WATER_BAND_1H
        return None

    def _fit_y(self) -> None:
        """The height for the visible window: fitted to what is shown, the
        water band left out, scaled by the gain around the baseline."""
        if self._xy is None or not self.isVisible():
            return
        x, y = self._xy
        (x0, x1), _yr = self._vb.viewRange()
        rng = M.y_range(x, y, x0, x1, exclude=self._exclude())
        if rng is None:
            return
        lo, hi = rng
        gain = max(1.0, float(self.ctx.scene.spectrum.y_gain))
        if gain > 1.0:
            vis = (x >= min(x0, x1)) & (x <= max(x0, x1))
            base = float(np.median(y[vis])) if vis.any() else 0.0
            lo = base - (base - lo) / gain
            hi = base + (hi - base) / gain
        self._vb.setYRange(lo, hi, padding=0.0)

    def _on_wheel(self, event, axis=None) -> None:
        """The wheel zooms the ppm axis; with Shift or Alt it scales the
        height (as the amplitude wheel does on traces)."""
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.AltModifier):
            delta = event.delta() if hasattr(event, "delta") else event.angleDelta().y()
            if delta:
                self.ctx.run("spectrum.gain", factor=1.25 if delta > 0 else 0.8)
            event.accept()
            return
        self._wheel_default(event, axis)

    def _draw_reference(self, sp, ppm_spec: np.ndarray, y: np.ndarray, src) -> None:
        """The water reference, scaled so its peak matches the spectrum's
        tallest point in the window that is fitted (never the water
        residual): the comparison is of shape and line width."""
        ref = src.reference if src is not None else None
        if not sp.reference or ref is None:
            self.ref_curve.setData([], [])
            return
        rfid = ref.select()
        ppm, _hz, spec = M.spectrum(rfid, ref.dwell, ref.spectrometer_mhz, ref.nucleus,
                                    line_broadening_hz=sp.lb_hz, phase_deg=sp.phase0,
                                    phase1_ms=sp.phase1_ms)
        ry = M.part(spec, sp.part)
        peak = float(np.nanmax(np.abs(ry))) or 1.0
        lo, hi = M.default_ppm_range(src.nucleus, ppm_spec)
        inside = (ppm_spec >= lo) & (ppm_spec <= hi)
        target = float(np.nanmax(np.abs(y[inside]))) if inside.any() else float(np.nanmax(np.abs(y)))
        scale = target / peak
        self.ref_curve.setPen(self._pg.mkPen(self.ctx.theme.token("warning", "#d29922"),
                                             width=1, style=Qt.PenStyle.DashLine))
        self.ref_curve.setData(ppm, ry * scale)

    def _show_metabolites(self, table) -> None:
        """Show one marker per entry of ``table`` (none for an empty one)."""
        pg = self._pg
        theme = self.ctx.theme
        while len(self._marker_pool) < len(table):
            # Centred ON the line. pyqtgraph's default puts a label beside
            # its line and flips the side at the middle of the view, so the
            # names either side of the middle pointed in opposite directions.
            line = pg.InfiniteLine(
                angle=90, movable=False, label="",
                labelOpts={"position": LABEL_ROWS[0], "movable": False,
                           "anchors": [(0.5, 0.5), (0.5, 0.5)]},
            )
            self.plot.addItem(line, ignoreBounds=True)
            self._marker_pool.append(line)
        pen = pg.mkPen(theme.dim, width=1, style=Qt.PenStyle.DashLine)
        for i, line in enumerate(self._marker_pool):
            line.setVisible(i < len(table))
        self._markers = self._marker_pool[: len(table)]
        self._marker_shifts = []
        for line, (name, shift, description) in zip(self._markers, table):
            line.setPos(shift)
            line.setPen(pen)
            line.label.setFormat(name)
            line.label.setColor(theme.text)
            from .. import fonts

            line.label.setFont(fonts.font(fonts.LABEL_PX))
            # The plot's own background behind the name, so the dashed line
            # does not run through the text.
            line.label.fill = pg.mkBrush(theme.plot_background)
            line.label.update()
            line.setToolTip(f"{name}: {description}, {shift:g} ppm")
            self._marker_shifts.append(float(shift))
        self._restagger()

    def _restagger(self, *_args) -> None:
        if not self._markers:
            return
        (x0, x1), _y = self._vb.viewRange()
        rows = M.stagger(self._marker_shifts, abs(x1 - x0), rows=len(LABEL_ROWS))
        for line, row in zip(self._markers, rows):
            label = getattr(line, "label", None)
            if label is not None:
                label.setPosition(LABEL_ROWS[min(row, len(LABEL_ROWS) - 1)])

    def _limit_pan(self, low: float, high: float) -> None:
        if not np.isfinite([low, high]).all():
            return
        margin = max(abs(high - low) * 0.02, 1e-9)
        self._vb.setLimits(xMin=min(low, high) - margin, xMax=max(low, high) + margin)

    # ------------------------------------------------------------------
    def reset_view(self) -> None:
        """The span the settings ask for (the standard window or the band),
        the height fitted to it, at the fitted height (gain 1)."""
        if self.ctx.scene.spectrum.y_gain != 1.0:
            self.ctx.run("spectrum.set", y_gain=1.0)
        if self.ctx.scene.spectrum.domain == "spectrum" and self._ppm_range:
            self.plot.getPlotItem().setXRange(*self._ppm_range, padding=0.02)
        elif self._xy is not None and self._xy[0].size:
            x = self._xy[0]
            self.plot.getPlotItem().setXRange(float(x.min()), float(x.max()), padding=0.02)
        self._fit_y()
        self._restagger()

    def fit_all(self) -> None:
        """The whole sampled band, water and lipid included."""
        sp = self.ctx.scene.spectrum
        if sp.domain == "spectrum" and sp.standard_window:
            self.ctx.run("spectrum.set", standard_window=False)
            return
        if self._xy is not None and self._xy[0].size:
            x = self._xy[0]
            self.plot.getPlotItem().setXRange(float(x.min()), float(x.max()), padding=0.02)
        self._fit_y()
        self._restagger()

    # ------------------------------------------------------------------
    def _on_hover(self, evt) -> None:
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        pt = self._vb.mapSceneToView(pos)
        fid = self.ctx.scene.spectrum.domain == "fid"
        unit, digits = ("s", 4) if fid else ("ppm", 2)
        self.readout = f"{pt.x():.{digits}f} {unit}   {pt.y():.3g}"
        self._vline.setPos(pt.x())
        self._hline.setPos(pt.y())
        self.ctx.readout.emit(self.readout)

    def _on_drag(self, event, axis=None) -> None:
        """Ctrl+drag phases (Shift for first order); otherwise pan as usual."""
        mods = event.modifiers()
        ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        if not ctrl or self.ctx.scene.spectrum.domain != "spectrum":
            self._drag_default(event, axis)
            return
        event.accept()
        first = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        x = event.pos().x()
        if event.isStart():
            self._phase_anchor = x
            return
        if self._phase_anchor is None:
            return
        dx = x - self._phase_anchor
        self._phase_anchor = x
        if first:
            self.ctx.run("spectrum.phase_drag", d_phase1_ms=dx * _PHASE1_PER_PX)
        else:
            self.ctx.run("spectrum.phase_drag", d_phase0=dx * _PHASE0_PER_PX)
        if event.isFinish():
            self._phase_anchor = None

    # -- queries (tests) ----------------------------------------------------
    def marker_rows(self) -> list[float]:
        return [getattr(line, "label").orthoPos for line in self._markers
                if getattr(line, "label", None) is not None]

    def marker_names(self) -> list[str]:
        return [line.label.format for line in self._markers
                if getattr(line, "label", None) is not None]


__all__ = ["LABEL_ROWS", "SpectrumCanvas"]
