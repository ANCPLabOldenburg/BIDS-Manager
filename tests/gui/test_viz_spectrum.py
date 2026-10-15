"""Spectra in the viewer: what would be wrong in a way somebody notices by
looking. Names drawn on top of each other, a drag that walks off the end of
the spectrum, a height that does not follow the zoom, an axis the wrong way.

The maths is tested Qt-free in ``tests/unit/test_viz_mrs.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nibabel")

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.gui.viz.bridge import SettingsHub  # noqa: E402
from tests.fixtures.signals import fid_at, write_mrs  # noqa: E402

pytestmark = pytest.mark.gui

#: Real chemical shifts, so the metabolite lines have something to sit
#: beside, which is what makes a label-overlap test mean anything.
_PEAKS = ((2.01, 1.0), (3.03, 0.8), (3.22, 0.5), (3.56, 0.3))


def _fid(points: int = 1024, dynamics: int = 4, *, seed: int = 0) -> np.ndarray:
    single = sum(fid_at(shift, points, decay_s=0.25, amplitude=a) for shift, a in _PEAKS)
    fid = np.stack([single] * dynamics, axis=1)
    noise = np.random.default_rng(seed).normal(scale=0.01, size=fid.shape)
    return fid + noise


@pytest.fixture()
def spectrum_file(tmp_path: Path) -> Path:
    return write_mrs(tmp_path / "sub-001" / "mrs", "sub-001_svs.nii.gz", _fid(),
                     {"EchoTime": 0.03, "RepetitionTime": 2.0})


def _viewer(qtbot, size=(1000, 600)) -> Viewer:
    """Built, SHOWN, and sized. Shown on purpose: a ``ViewBox`` with no
    geometry defers its range update, so an unshown canvas reports the range
    it had before the zoom and every autoscaling assertion would pass by
    accident."""
    viewer = Viewer(kind="spectrum")
    qtbot.addWidget(viewer)
    viewer.resize(*size)
    viewer.show()
    qtbot.waitExposed(viewer)
    return viewer


def _open(qtbot, path: Path, root: Path) -> Viewer:
    viewer = _viewer(qtbot)
    with qtbot.waitSignal(viewer.loaded, timeout=30_000):
        viewer.set_file(path, root)
    viewer.qstore.flush()
    return viewer


def _canvas(viewer):
    return viewer.presenter.canvas


def _view(viewer):
    return _canvas(viewer).plot.getPlotItem().getViewBox()


def _x_span(viewer) -> float:
    return float(np.ptp(_view(viewer).viewRange()[0]))


class TestItOpens:
    def test_a_spectrum_is_read_and_drawn(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        assert viewer.page() == "content"
        assert viewer.source().nucleus == "1H"
        x, y = _canvas(viewer).curve.getData()
        assert x.size == 1024 and np.isfinite(y).all()
        assert "1H spectrum" in viewer.summary_text()

    def test_the_strongest_peak_is_where_it_was_put(self, qtbot, spectrum_file, tmp_path):
        """The ppm axis drawn the wrong way round is a plausible picture with
        every metabolite on the wrong side of the water."""
        viewer = _open(qtbot, spectrum_file, tmp_path)
        x, y = _canvas(viewer).curve.getData()
        assert x[int(np.argmax(y))] == pytest.approx(2.01, abs=0.03)
        assert _view(viewer).xInverted(), "chemical shift runs right to left"

    def test_a_file_with_no_mrs_header_says_so(self, qtbot, tmp_path):
        """Not an exception: a NIfTI in ``mrs/`` without the header is a real
        thing a converter can produce."""
        import nibabel as nib

        folder = tmp_path / "sub-001" / "mrs"
        folder.mkdir(parents=True)
        path = folder / "sub-001_svs.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4)),
                 str(path))
        viewer = _viewer(qtbot)
        with qtbot.waitSignal(viewer.load_failed, timeout=30_000) as failed:
            viewer.set_file(path, tmp_path)
        assert "no NIfTI-MRS header" in failed.args[1]
        assert viewer.source() is None


class TestTheMetaboliteLabels:
    def test_they_are_drawn_with_names(self, qtbot, spectrum_file, tmp_path):
        canvas = _canvas(_open(qtbot, spectrum_file, tmp_path))
        assert {"NAA", "Cr", "Cho", "mI"} <= set(canvas.marker_names())
        assert canvas._marker_shifts == sorted(canvas._marker_shifts)

    def test_crowded_names_are_moved_apart(self, qtbot, spectrum_file, tmp_path):
        """Creatine at 3.03 and choline at 3.22 are a fifth of a ppm apart.
        On one line, across the whole window, they overlap."""
        canvas = _canvas(_open(qtbot, spectrum_file, tmp_path))
        canvas.reset_view()
        assert len(set(canvas.marker_rows())) > 1, "all on one row means they overlap"

    def test_zooming_in_gives_them_room_back(self, qtbot, spectrum_file, tmp_path):
        """A label's width in pixels does not change with the zoom."""
        from bidsmgr.gui.viz.canvases.spectrum import LABEL_ROWS

        viewer = _open(qtbot, spectrum_file, tmp_path)
        _canvas(viewer).plot.getPlotItem().setXRange(2.9, 3.4, padding=0)
        assert set(_canvas(viewer).marker_rows()) == {LABEL_ROWS[0]}

    def test_a_row_is_never_off_the_bottom(self, qtbot, spectrum_file, tmp_path):
        from bidsmgr.gui.viz.canvases.spectrum import LABEL_ROWS

        viewer = _open(qtbot, spectrum_file, tmp_path)
        _canvas(viewer).plot.getPlotItem().setXRange(3.0, 3.05, padding=0)
        assert all(row in LABEL_ROWS for row in _canvas(viewer).marker_rows())

    def test_turning_them_off_removes_them(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        assert viewer.action("spectrum.metabolites").isChecked()
        viewer.trigger("spectrum.metabolites")
        viewer.qstore.flush()
        assert _canvas(viewer).marker_names() == []
        assert not viewer.action("spectrum.metabolites").isChecked()

    def test_another_nucleus_gets_no_1h_labels(self, qtbot, tmp_path):
        path = write_mrs(tmp_path / "sub-001" / "mrs", "sub-001_svs.nii.gz", _fid(),
                         {"ResonantNucleus": ["31P"], "SpectrometerFrequency": [49.9]})
        viewer = _open(qtbot, path, tmp_path)
        assert _canvas(viewer).marker_names() == []
        assert not viewer.action("spectrum.metabolites").isEnabled()


class TestTheView:
    def test_panning_cannot_leave_the_data(self, qtbot, spectrum_file, tmp_path):
        """Dragging used to walk into an empty pane with no way back but
        Reset, which is the complaint that produced the limits."""
        viewer = _open(qtbot, spectrum_file, tmp_path)
        view = _view(viewer)
        _canvas(viewer).reset_view()
        width = _x_span(viewer)
        for _ in range(60):
            view.translateBy(x=+2.0)
        low, high = view.state["limits"]["xLimits"]
        assert view.viewRange()[0][1] <= high + 1e-6
        for _ in range(120):
            view.translateBy(x=-2.0)
        assert view.viewRange()[0][0] >= low - 1e-6
        assert _x_span(viewer) == pytest.approx(width)

    def test_the_height_follows_the_visible_slice(self, qtbot, spectrum_file, tmp_path):
        """Zoomed past the big peak, the small ones should fill the pane."""
        viewer = _open(qtbot, spectrum_file, tmp_path)
        view = _view(viewer)
        _canvas(viewer).reset_view()
        whole = np.ptp(view.viewRange()[1])
        _canvas(viewer).plot.getPlotItem().setXRange(3.4, 4.0, padding=0)
        assert np.ptp(view.viewRange()[1]) < whole
        # ...and after a wheel or a drag too: pyqtgraph's own auto-range
        # switched itself off at the first gesture.
        view.scaleBy((0.5, 1.0))
        view.translateBy((0.1, 0.0))
        x0, x1 = view.viewRange()[0]
        canvas = _canvas(viewer)
        x, y = canvas._xy
        lo, hi = view.viewRange()[1]
        vis = (x >= min(x0, x1)) & (x <= max(x0, x1))
        assert lo <= y[vis].min() and hi >= y[vis].max()

    def test_the_water_runs_off_the_top_instead_of_flattening_the_rest(
            self, qtbot, tmp_path):
        folder = tmp_path / "sub-001" / "mrs"
        fid = fid_at(4.65, 1024, amplitude=50.0) + fid_at(2.01, 1024)
        path = write_mrs(folder, "sub-001_svs.nii.gz", fid)
        viewer = _open(qtbot, path, tmp_path)
        canvas, view = _canvas(viewer), _view(viewer)
        canvas.plot.getPlotItem().setXRange(1.5, 5.0, padding=0)
        with_water_out = np.ptp(view.viewRange()[1])
        viewer.trigger("spectrum.water")
        viewer.qstore.flush()
        assert np.ptp(view.viewRange()[1]) > 5 * with_water_out

    def test_the_height_can_be_scaled_to_look_under_the_peaks(self, qtbot, spectrum_file,
                                                                tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        view = _view(viewer)
        fitted = np.ptp(view.viewRange()[1])
        viewer.trigger("spectrum.gain_up")
        viewer.trigger("spectrum.gain_up")
        viewer.qstore.flush()
        assert np.ptp(view.viewRange()[1]) == pytest.approx(fitted / 1.25 ** 2, rel=1e-3)
        viewer.trigger("spectrum.reset")
        viewer.qstore.flush()
        assert viewer.scene.spectrum.y_gain == 1.0

    def test_a_file_opens_phased_with_its_quality_in_the_footer(self, qtbot, spectrum_file,
                                                                 tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        assert viewer.scene.spectrum.part == "real"
        text = viewer.presenter.summary()
        assert "SNR" in text and "ppm" in text

    def test_fit_all_is_wider_than_the_standard_window(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        standard = _x_span(viewer)
        viewer.trigger("spectrum.fit")
        viewer.qstore.flush()
        assert _x_span(viewer) > standard
        assert not viewer.action("spectrum.window").isChecked(), (
            "fitting the band is leaving the standard window, and says so")

    def test_the_standard_window_actually_moves_the_view(self, qtbot, spectrum_file, tmp_path):
        """It used to recompute the span and leave the view where it was."""
        viewer = _open(qtbot, spectrum_file, tmp_path)
        clipped = _x_span(viewer)
        viewer.trigger("spectrum.window")
        viewer.qstore.flush()
        assert _x_span(viewer) > clipped
        viewer.trigger("spectrum.window")
        viewer.qstore.flush()
        assert _x_span(viewer) == pytest.approx(clipped, rel=0.05)

    def test_the_fid_is_drawn_against_seconds(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        viewer.trigger("spectrum.fid")
        viewer.qstore.flush()
        pi = _canvas(viewer).plot.getPlotItem()
        assert "Time" in pi.getAxis("bottom").labelText
        assert not _view(viewer).xInverted()
        assert viewer.presenter.domain.currentData() == "fid", "the combo follows"
        viewer.trigger("spectrum.fid")
        viewer.qstore.flush()
        assert _view(viewer).xInverted()


class TestProcessing:
    def test_the_repeats_can_be_stepped_through(self, qtbot, tmp_path):
        """How a corrupted repeat is found."""
        fid = _fid(dynamics=3)
        fid[:, 2] *= 10.0
        path = write_mrs(tmp_path / "sub-001" / "mrs", "sub-001_svs.nii.gz", fid)
        viewer = _open(qtbot, path, tmp_path)
        p = viewer.presenter
        assert p.repeat.isEnabled() and p.repeat.maximum() == 3
        averaged = float(np.max(_canvas(viewer).curve.getData()[1]))
        p.repeat.setValue(3)            # the third repeat, 1-based on screen
        viewer.qstore.flush()
        assert viewer.scene.spectrum.repeat == 2
        assert float(np.max(_canvas(viewer).curve.getData()[1])) > 2.0 * averaged

    def test_edit_conditions_offer_the_difference(self, qtbot, tmp_path):
        base = sum(fid_at(s, 1024, amplitude=a) for s, a in _PEAKS)
        on_off = np.stack([base + fid_at(3.0, 1024, amplitude=0.4), base], axis=1)
        path = write_mrs(tmp_path / "sub-001" / "mrs", "sub-001_svs.nii.gz", on_off,
                         {"dim_5": "DIM_EDIT"})
        viewer = _open(qtbot, path, tmp_path)
        p = viewer.presenter
        assert p.edit.isVisibleTo(viewer)
        assert p.edit.findData(-2) >= 0
        p.edit.setCurrentIndex(p.edit.findData(-2))
        viewer.qstore.flush()
        x, y = _canvas(viewer).curve.getData()
        assert x[int(np.argmax(y))] == pytest.approx(3.0, abs=0.03), (
            "only the edited resonance survives the subtraction")

    def test_a_plain_file_hides_the_edit_control(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        assert not viewer.presenter.edit.isVisibleTo(viewer)

    def test_line_broadening_lowers_the_peak(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        before = float(np.max(_canvas(viewer).curve.getData()[1]))
        viewer.presenter.lb.type_value(10.0)
        viewer.qstore.flush()
        assert float(np.max(_canvas(viewer).curve.getData()[1])) < before

    def test_a_phase_drag_turns_the_phase_and_reset_undoes_it(
        self, qtbot, spectrum_file, tmp_path,
    ):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        start = viewer.scene.spectrum.phase0          # phased automatically on open
        viewer.run("spectrum.phase_drag", d_phase0=30.0)
        viewer.run("spectrum.phase_drag", d_phase1_ms=0.25)
        viewer.qstore.flush()
        sp = viewer.scene.spectrum
        assert (sp.phase0, sp.phase1_ms) == (pytest.approx(start + 30.0), 0.25)
        assert viewer.presenter.phase0.value() == pytest.approx(start + 30.0), \
            "the slider and its number follow"
        viewer.trigger("spectrum.reset_processing")
        viewer.qstore.flush()
        sp = viewer.scene.spectrum
        assert (sp.phase0, sp.phase1_ms, sp.lb_hz) == (0.0, 0.0, 2.0)

    def test_the_water_reference_is_offered_only_when_there_is_one(self, qtbot, tmp_path):
        folder = tmp_path / "sub-001" / "mrs"
        lone = write_mrs(folder, "sub-001_acq-a_svs.nii.gz", _fid())
        viewer = _open(qtbot, lone, tmp_path)
        assert not viewer.action("spectrum.reference").isEnabled()

        write_mrs(folder, "sub-001_acq-a_mrsref.nii.gz", fid_at(4.65, 1024))
        viewer = _open(qtbot, lone, tmp_path)
        assert viewer.action("spectrum.reference").isEnabled()
        viewer.trigger("spectrum.reference")
        viewer.qstore.flush()
        x, y = _canvas(viewer).ref_curve.getData()
        assert x is not None and x.size == 1024
        assert np.max(np.abs(y)) == pytest.approx(
            np.max(np.abs(_canvas(viewer).curve.getData()[1])), rel=1e-6), (
            "scaled to the spectrum: compared by shape, not amplitude")

    def test_the_header_is_one_click_away(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        viewer.presenter.show_header()
        dlg = viewer.presenter.header_dialog
        qtbot.addWidget(dlg)
        from PyQt6.QtWidgets import QPlainTextEdit

        body = dlg.findChild(QPlainTextEdit).toPlainText()
        assert '"EchoTime": 0.03' in body
        assert '"SpectrometerFrequency"' in body


class TestTheSharedControls:
    def test_the_default_line_is_not_a_hairline(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        assert _canvas(viewer).curve.opts["pen"].width() >= 2

    def test_the_line_settings_change_the_trace(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)

        def keep(s):
            s.traces.line_width = 6
            s.traces.line_color = "#ff7b72"

        SettingsHub.instance().update(keep)
        pen = _canvas(viewer).curve.opts["pen"]
        assert pen.width() == 6
        assert pen.color().name() == "#ff7b72"

    def test_one_trace_means_no_colour_by_type(self, qtbot, spectrum_file, tmp_path,
                                               monkeypatch):
        from bidsmgr.gui.viz.panels.line_style import LineStyleDialog

        monkeypatch.setattr(LineStyleDialog, "exec", lambda self: 0)
        viewer = _open(qtbot, spectrum_file, tmp_path)
        viewer.trigger("traces.line")
        dlg = viewer.presenter.line_dialog
        qtbot.addWidget(dlg)
        assert not dlg._by_type.isEnabled()
        assert dlg._one_colour.isChecked()

    def test_the_crosshair_reads_out_ppm(self, qtbot, spectrum_file, tmp_path):
        from PyQt6.QtCore import QPointF

        viewer = _open(qtbot, spectrum_file, tmp_path)
        canvas = _canvas(viewer)
        canvas._proxy.sigDelayed.emit((QPointF(canvas.plot.sceneBoundingRect().center()),))
        assert "ppm" in viewer.readout_text()

    def test_the_crosshair_survives_a_redraw(self, qtbot, spectrum_file, tmp_path):
        viewer = _open(qtbot, spectrum_file, tmp_path)
        viewer.presenter.lb.type_value(5.0)
        viewer.qstore.flush()
        items = _canvas(viewer).plot.getPlotItem().items
        assert _canvas(viewer)._vline in items
        assert _canvas(viewer)._hline in items


class TestTheme:
    def test_a_swap_while_open_repaints_the_plot(self, qtbot, spectrum_file, tmp_path):
        """pyqtgraph reads no QSS, so the palette has to be handed down."""
        from PyQt6.QtGui import QColor

        from bidsmgr.gui import theme_manager
        from bidsmgr.gui.viz.bridge import ThemeHub

        viewer = _open(qtbot, spectrum_file, tmp_path)
        canvas = _canvas(viewer)
        before = canvas.plot.backgroundBrush().color().name()
        try:
            viewer.repaint_for_palette(theme_manager.LIGHT)
            assert canvas.plot.backgroundBrush().color().name() != before
            assert canvas._vline.pen.color() == QColor(viewer.presenter.ctx.theme.dim), (
                "the crosshair follows the palette too")
            assert canvas.marker_names(), "the labels survive a repaint"
        finally:
            ThemeHub.instance().publish(theme_manager.DARK)


class TestTheEditor:
    def test_an_mrs_file_routes_to_the_spectrum_viewer(self, qtbot, spectrum_file, tmp_path):
        from bidsmgr.gui.editor_panel import EditorPanel

        ep = EditorPanel()
        qtbot.addWidget(ep)
        ep._set_root(tmp_path, persist=False)
        with qtbot.waitSignal(ep._spectrum_viewer.loaded, timeout=30_000):
            ep._on_file_selected(spectrum_file)
        assert ep._center_stack.currentWidget() is ep._spectrum_viewer
        assert ep._nifti_viewer.current_file() is None, "never sent to the volume viewer"

    def test_selecting_something_else_clears_it(self, qtbot, spectrum_file, tmp_path):
        """The MRS pane used to keep its file after any selection but a NIfTI."""
        from bidsmgr.gui.editor_panel import EditorPanel

        sidecar = spectrum_file.with_name("sub-001_svs.json")
        sidecar.write_text('{"EchoTime": 0.03}', encoding="utf-8")
        ep = EditorPanel()
        qtbot.addWidget(ep)
        ep._set_root(tmp_path, persist=False)
        with qtbot.waitSignal(ep._spectrum_viewer.loaded, timeout=30_000):
            ep._on_file_selected(spectrum_file)
        ep._on_file_selected(sidecar)
        assert ep._spectrum_viewer.current_file() is None
        assert ep._spectrum_viewer.source() is None


def test_the_options_persist_to_a_new_window_but_not_the_phase(qtbot, spectrum_file):
    from bidsmgr.gui.viz.bridge import SettingsHub

    first = _open(qtbot, spectrum_file, spectrum_file.parents[2])
    first.run("spectrum.set", lb_hz=6.0, part="magnitude")
    first.run("spectrum.toggle", field="metabolites")
    first.qstore.flush()
    first.presenter.stop()
    SettingsHub.reset_instance()
    second = _open(qtbot, spectrum_file, spectrum_file.parents[2])
    sp = second.scene.spectrum
    assert (sp.lb_hz, sp.part, sp.metabolites) == (6.0, "magnitude", False)
