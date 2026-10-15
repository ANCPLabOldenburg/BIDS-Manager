"""Spectra (NIfTI-MRS) in the viewer shell.

An MRS file is a NIfTI by container only; its data block is a complex FID,
and showing it as slices would show nothing. So ``mrs/`` opens here: the
spectrum in ppm, or the FID, with line broadening, zero- and first-order
phase, the repeats averaged or one at a time, the edit conditions picked or
subtracted, labelled metabolites, the conventional window, the water
reference overlaid, and the NIfTI-MRS header itself (extension 44) one click
away. The read and the transform run on a QThread: this ends in an FFT.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ....viz import keynames
from ....viz.data.spectrum import SpectrumSource, read_mrs
from ....viz.scene import Scene, SourceRef, SpectrumLayer
from ..context import ViewerContext

log = logging.getLogger(__name__)

#: Two rows, by purpose: how the signal is PROCESSED (domain, component,
#: phase, line broadening, which repeats), then what is SHOWN over it and how
#: the view is framed, with the file's header and its voxel on the anatomy.
#: ONE row, by purpose: what is shown (the FID) and phasing it; the marks
#: read against; the view; the file's context (its header, its voxel on
#: the anatomy). Everything else is in the controls column
#: (``panels.spectrum_controls``).
TOOLBAR_ROWS = (
    ("spectrum.fid", "spectrum.auto_phase", "|", "spectrum.metabolites", "spectrum.window",
     "|", "spectrum.fit", "spectrum.reset", "|", "widget:header", "spectrum.anatomy",
     "stretch", "help.shortcuts"),
)

_PARTS = (("magnitude", "Magnitude"), ("real", "Real"), ("imaginary", "Imaginary"),
          ("phase", "Phase"))


class SpectrumPresenter:
    kind = "spectrum"
    title = "Spectroscopy"
    empty_hint = "Select a spectroscopy file."
    mouse_canvases = ()

    def __init__(self, viewer, ctx: ViewerContext) -> None:
        self.viewer = viewer
        self.ctx = ctx
        self.source: Optional[SpectrumSource] = None
        self._generation = 0
        self._path: Optional[Path] = None
        from ..canvases.spectrum import SpectrumCanvas

        self.canvas = SpectrumCanvas(ctx)
        from ..panels.side_column import SideColumn
        from ..panels.spectrum_controls import SpectrumControls

        self._syncing = False
        self.controls = SpectrumControls(self)

        def remember(on: bool) -> None:
            if ctx.settings.spectrum_controls != on:
                ctx.settings_hub.update(lambda st: setattr(st, "spectrum_controls", on))

        self.column = SideColumn(self.canvas, self.controls, remember=remember,
                                 changed=viewer.refresh_actions)
        self.content = self.column.widget
        ctx.jobs.done.connect(self._on_job_done)
        ctx.jobs.failed.connect(self._on_job_failed)
        ctx.qstore.changed.connect(lambda _p: (self.sync_widgets(), viewer.update_footer()))
        ctx.qstore.changed.connect(self._maybe_remember)
        self._first_file = True
        from PyQt6.QtCore import QTimer

        self._memory_timer = QTimer(viewer)
        self._memory_timer.setSingleShot(True)
        self._memory_timer.setInterval(400)
        self._memory_timer.timeout.connect(self._remember)

    # ------------------------------------------------------------------
    def toolbar_rows(self):
        return TOOLBAR_ROWS

    def make_widget(self, name: str):
        return {"header": self._widget_header}.get(name, lambda: None)()

    # -- the controls column ---------------------------------------------------

    def inspector_open(self) -> bool:
        return self.column.is_open()

    def set_inspector(self, on: bool, *, remember: bool = True) -> None:
        self.column.set_open(on, remember=remember)

    def _make_processing(self) -> None:
        """The processing controls (placed by the controls column)."""
        self.domain = QComboBox()
        self.domain.addItem("Spectrum", "spectrum")
        self.domain.addItem("FID (time domain)", "fid")
        self.domain.currentIndexChanged.connect(
            lambda _i: self._run_ui("spectrum.set", domain=self.domain.currentData()))
        self.part = QComboBox()
        for value, text in _PARTS:
            self.part.addItem(text, value)
        self.part.currentIndexChanged.connect(
            lambda _i: self._run_ui("spectrum.set", part=self.part.currentData()))
        from ..controls import NumberControl

        def number(lo, hi, step, unit, default, key):
            nc = NumberControl(lo, hi, step=step, unit=unit, default=default)
            nc.slider.setMinimumWidth(90)
            nc.value_changed.connect(lambda v: self._run_ui("spectrum.set", **{key: float(v)}))
            # A drag is one undo step, and phasing by slider is live.
            nc.pressed.connect(self.ctx.store.begin_gesture)
            nc.released.connect(self.ctx.store.end_gesture)
            return nc

        self.lb = number(0.0, 20.0, 0.5, "Hz", 0.0, "lb_hz")
        self.phase0 = number(-180.0, 180.0, 0.5, "deg", 0.0, "phase0")
        self.phase1 = number(-2.0, 2.0, 0.01, "ms", 0.0, "phase1_ms")
        self.repeat = QSpinBox()
        self.repeat.setRange(0, 0)
        self.repeat.setSpecialValueText("Averaged")
        self.repeat.setKeyboardTracking(False)
        self.repeat.valueChanged.connect(lambda v: self._run_ui("spectrum.set", repeat=int(v) - 1))
        self.edit = QComboBox()
        self.edit.currentIndexChanged.connect(
            lambda _i: self._run_ui("spectrum.set", edit=self.edit.currentData()))
        #: What each control does, shown on hover over it and its label.
        self.tips = {
            "domain": "The spectrum is what is read; the FID is what the scanner measured, "
                      "and is what to look at when a spectrum looks wrong.",
            "part": "The real part, phased, is how every MRS package shows a spectrum. "
                    "Magnitude needs no phase, but its water tail is about fourteen times "
                    "the real part's at 4.2 ppm.",
            "lb": "An exponential on the FID, for viewing: resolution traded for a smoother "
                  "line. Fitting packages apply none in vivo; the SNR and line width under "
                  "QC are measured without it.",
            "phase0": "Zero-order phase. Auto phase (P) sets it from NAA, creatine and "
                      f"choline; drag the slider, or {keynames.mouse(['ctrl'], 'drag')} "
                      "the spectrum.",
            "phase1": "First-order (frequency-dependent) phase, as the acquisition delay it "
                      f"undoes. {keynames.mouse(['ctrl', 'shift'], 'drag')} the spectrum "
                      "to set it by eye.",
            "repeat": "Every repeat (transient) averaged, or one at a time: how a corrupted "
                      "repeat, a motion or frequency drift, is found.",
            "edit": "Spectral editing (MEGA-PRESS and similar) acquires an edit-on and an "
                    "edit-off condition: their DIFFERENCE is what the scan was run for "
                    "(GABA, GSH).",
        }

    def _widget_header(self) -> QWidget:
        btn = QPushButton("Header")
        btn.setObjectName("tb-btn")
        btn.setToolTip("The NIfTI-MRS header inside the file (extension 44): "
                       "every acquisition parameter the spectrum depends on")
        btn.clicked.connect(lambda _c=False: self.show_header())
        return btn

    # ------------------------------------------------------------------
    def load(self, path: Path, root: Optional[Path]) -> None:
        if self._memory_timer.isActive():
            self._memory_timer.stop()
            self._remember()
        self._generation += 1
        self._path = Path(path)
        self.ctx.jobs.start("read", self._generation, read_mrs, self._path)

    def clear(self) -> None:
        self._generation += 1
        self.ctx.jobs.cancel("read")
        self.source = None
        self.ctx.store.sources.clear()
        self.ctx.store.replace_scene(Scene())
        self.ctx.qstore.flush()

    def stop(self) -> None:
        if self._memory_timer.isActive():
            self._memory_timer.stop()
            self._remember()

    def _maybe_remember(self, paths) -> None:
        if self.source is not None and any(p.startswith("spectrum") for p in paths):
            self._memory_timer.start()

    def _remember(self) -> None:
        """The processing and display options, for the next file and window
        (``viz.memory``; never the phase, which is the file's)."""
        from ....viz import memory

        prefs = memory.spectrum_prefs(self.ctx.scene.spectrum)
        self.ctx.settings_hub.update(lambda s: setattr(s, "spectrum_state", prefs))

    def _on_job_done(self, tag: str, generation: int, result) -> None:
        if generation != self._generation or tag != "read":
            return
        if result is None:
            self.viewer.on_load_failed(
                self._path,
                f"{self._path.name} is in an mrs/ folder but carries no NIfTI-MRS "
                "header, so there is no spectrum to show. The acquisition parameters "
                "a spectrum needs (the dwell time, the spectrometer frequency and the "
                "nucleus) live in that header.")
            return
        self.source = result
        store = self.ctx.store
        keep = store.scene.spectrum.model_copy()
        if self._first_file:
            # The options as last left, in any window.
            from ....viz import memory

            keep = memory.restore_spectrum(keep, self.ctx.settings.spectrum_state)
            self._first_file = False
        scene = Scene()
        scene.sources = {"spec0": SourceRef(id="spec0", path=str(result.path), kind="spectrum")}
        scene.layers = [SpectrumLayer(id="spec", source="spec0", name=result.path.name)]
        # The look carries to the next file; what depends on THIS file does
        # not: the selection, the height gain, and the phase, which is
        # found again for this file.
        keep.repeat, keep.edit = -1, -1
        keep.phase0, keep.phase1_ms, keep.y_gain = 0.0, 0.0, 1.0
        keep.reference = keep.reference and result.reference is not None
        scene.spectrum = keep
        store.sources = {"spec0": result}
        store.replace_scene(scene)
        if keep.part != "magnitude":
            self.ctx.run("spectrum.auto_phase")
        self._qc_key = None
        self.viewer.on_loaded(result.path)
        self.ctx.qstore.flush()

    def _on_job_failed(self, tag: str, generation: int, message: str) -> None:
        if generation == self._generation and tag == "read":
            self.viewer.on_load_failed(self._path, message)

    # ------------------------------------------------------------------
    _syncing = False

    def _run_ui(self, command_id: str, **params) -> None:
        if not self._syncing:
            self.ctx.run(command_id, **params)

    def sync_widgets(self) -> None:
        if not hasattr(self, "domain"):
            return
        sp = self.ctx.scene.spectrum
        src = self.source
        self._syncing = True
        try:
            self.domain.setCurrentIndex(self.domain.findData(sp.domain))
            self.part.setCurrentIndex(self.part.findData(sp.part))
            self.lb.set_value(sp.lb_hz)
            self.phase0.set_value(sp.phase0)
            self.phase1.set_value(sp.phase1_ms)
            n = src.n_dynamics if src is not None else 1
            self.repeat.setRange(0, max(0, n))
            self.repeat.setEnabled(n > 1)
            self.repeat.setValue(sp.repeat + 1)
            edits = src.n_edits if src is not None else 1
            wanted = [(-1, "Average")] + [(i, f"Condition {i + 1}") for i in range(edits)]
            if edits >= 2:
                wanted.append((-2, "Difference (1 - 2)"))
            have = [self.edit.itemData(i) for i in range(self.edit.count())]
            if have != [v for v, _t in wanted]:
                self.edit.clear()
                for value, text in wanted:
                    self.edit.addItem(text, value)
            self.edit.setCurrentIndex(max(0, self.edit.findData(sp.edit)))
            self.controls.sync(edits)
        finally:
            self._syncing = False

    def action_context(self) -> dict[str, Any]:
        src = self.source
        sp = self.ctx.scene.spectrum
        return {
            "spectrum": src is not None,
            "proton": bool(src and src.nucleus.upper() == "1H"),
            "reference": bool(src and src.reference is not None),
            "spectrum.fid": sp.domain == "fid",
            "spectrum.metabolites": sp.metabolites,
            "spectrum.window": sp.standard_window,
            "spectrum.exclude_water": sp.exclude_water,
            "spectrum.reference": sp.reference,
            "voxel": bool(src and src.affine is not None),
            "panel.inspector": self.inspector_open(),
        }

    def gui_action(self, name: str, params) -> bool:
        if name == "inspector":
            self.set_inspector(not self.inspector_open())
            return True
        if name == "spectrum_fit":
            self.canvas.fit_all()
            return True
        if name == "spectrum_reset":
            self.canvas.reset_view()
            return True
        if name == "anatomy":
            self.show_on_anatomy()
            return True
        if name == "line":
            from ..panels.line_style import LineStyleDialog

            ts = self.ctx.settings.traces
            dlg = LineStyleDialog(ts.line_width, ts.line_color or None,
                                  allow_by_type=False, channel_types=(),
                                  parent=self.viewer)
            self.line_dialog = dlg
            dlg.exec()
            return True
        return False

    def show_on_anatomy(self) -> None:
        """The voxel the spectrum came from, outlined on the subject's own
        anatomical image (found through the dataset, same session first)."""
        from PyQt6.QtCore import Qt as _Qt

        from ....viz.bids import anatomical_for
        from ..viewer import Viewer

        src = self.source
        if src is None:
            return
        anat = anatomical_for(src.path)
        if anat is None:
            self.viewer.status_message.emit(
                f"No anatomical image found for {src.path.name}: looked in this "
                "subject's anat/ folders for a T1w, T2w or FLAIR.")
            return
        dlg = QDialog(self.viewer)
        dlg.setAttribute(_Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.setWindowTitle(f"{src.path.name} on {anat.name}")
        dlg.resize(1000, 640)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(0, 0, 0, 0)
        viewer = Viewer(kind="volume", parent=dlg)
        v.addWidget(viewer)
        centre = src.voxel_centre_world()

        def on_overlay(_layer_id: str) -> None:
            if centre is not None:
                viewer.run("cursor.set_world", x=float(centre[0]), y=float(centre[1]),
                           z=float(centre[2]))

        viewer.overlay_added.connect(on_overlay)
        viewer.status_message.connect(self.viewer.status_message)
        dlg.finished.connect(lambda _r: viewer.stop_loading())
        viewer.set_file(anat, self.viewer.current_root())
        viewer.add_overlay(src.path)
        self.anatomy_dialog = dlg
        self.anatomy_viewer = viewer
        dlg.show()

    def show_header(self, rule: str = "") -> None:
        """The NIfTI-MRS header (``rule`` is accepted for the shared viewer
        API; a spectrum's header has no rows to select)."""
        del rule
        src = self.source
        if src is None:
            return
        dlg = QDialog(self.viewer)
        dlg.setWindowTitle(f"NIfTI-MRS header: {src.path.name}")
        dlg.resize(560, 600)
        v = QVBoxLayout(dlg)
        text = QPlainTextEdit(json.dumps(src.header, indent=2, sort_keys=True))
        text.setReadOnly(True)
        v.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        self.header_dialog = dlg
        dlg.show()

    def apply_mode(self) -> None:
        pass

    def apply_graph(self) -> None:
        pass

    def restore_panels(self) -> None:
        if self.ctx.settings.spectrum_controls and not self.inspector_open():
            self.set_inspector(True, remember=False)

    def canvases(self, kind: str) -> list:
        return [self.canvas] if kind == "spectrum" and self.canvas.isVisible() else []

    def figure_widget(self) -> QWidget:
        return self.canvas

    _qc_key = None
    _qc_text = ""

    def quality(self) -> str:
        """SNR, NAA line width and position of the spectrum as phased, before
        any line broadening (cached per selection and phase)."""
        from ....viz.compute import mrs as M

        src = self.source
        if src is None:
            return ""
        sp = self.ctx.scene.spectrum
        key = (id(src), sp.repeat, sp.edit, sp.phase0, sp.phase1_ms)
        if key != self._qc_key:
            fid = src.select(dynamic=sp.repeat, edit=sp.edit)
            ppm, _hz, spec = M.spectrum(fid, src.dwell, src.spectrometer_mhz, src.nucleus,
                                        phase_deg=sp.phase0, phase1_ms=sp.phase1_ms)
            self._qc_text = M.describe_qc(M.qc_metrics(ppm, spec, src.spectrometer_mhz,
                                                       src.nucleus))
            self._qc_key = key
        return self._qc_text

    def summary(self) -> str:
        if self.source is None:
            return ""
        text = self.source.describe()
        qc = self.quality()
        sp = self.ctx.scene.spectrum
        if sp.lb_hz:
            text += f" · LB {sp.lb_hz:g} Hz"
        return f"{text} · {qc}" if qc else text

    def readout(self) -> str:
        return self.canvas.readout


__all__ = ["SpectrumPresenter"]
