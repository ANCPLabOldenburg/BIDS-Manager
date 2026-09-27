"""The spectroscopy pane: its controls, and the ones it shares.

The maths is tested Qt-free in ``tests/unit/test_mrs_spectrum.py``. These are
the things that would be wrong in a way somebody would only notice by
looking: names drawn on top of each other, a drag that walks off the end of
the spectrum, a height that does not follow the zoom.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gui

#: What a NIfTI-MRS header lives in.
_MRS_CODE = 44


def _write_mrs(
    root: Path,
    name: str = "sub-001_svs.nii.gz",
    *,
    points: int = 1024,
    dynamics: int = 4,
    nucleus: str = "1H",
) -> Path:
    """A minimal but genuine NIfTI-MRS file: complex FID, header in code 44.

    Peaks are placed at real chemical shifts so the metabolite lines have
    something to sit beside, which is what makes a label-overlap test mean
    anything.
    """
    import nibabel as nib

    folder = root / "sub-001" / "mrs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name

    dwell = 1.0 / 2000.0          # 2 kHz sweep
    mhz = 123.26                  # 1H at 2.9 T
    reference = 4.65
    t = np.arange(points) * dwell
    fid = np.zeros((points, dynamics), dtype=np.complex128)
    for shift, amplitude in ((2.01, 1.0), (3.03, 0.8), (3.22, 0.5), (3.56, 0.3)):
        hz = -(shift - reference) * mhz
        decay = np.exp(-t / 0.25)
        for d in range(dynamics):
            fid[:, d] += amplitude * decay * np.exp(2j * np.pi * hz * t)
    rng = np.random.default_rng(0)
    fid += rng.normal(scale=0.01, size=fid.shape)

    data = fid.reshape(1, 1, 1, points, dynamics)
    affine = np.eye(4)
    img = nib.Nifti2Image(data, affine)
    img.header.set_zooms((1.0, 1.0, 1.0, dwell, 1.0))
    header = {
        "SpectrometerFrequency": [mhz],
        "ResonantNucleus": [nucleus],
        "DwellTime": dwell,
        "EchoTime": 0.03,
        "RepetitionTime": 2.0,
    }
    img.header.extensions.append(
        nib.nifti1.Nifti1Extension(_MRS_CODE, json.dumps(header).encode())
    )
    nib.save(img, str(path))
    (folder / name.replace(".nii.gz", ".json")).write_text(
        json.dumps({"EchoTime": 0.03, "RepetitionTime": 2.0}), encoding="utf-8",
    )
    return path


@pytest.fixture()
def spectrum_file(tmp_path: Path) -> Path:
    return _write_mrs(tmp_path)


def _pane(qtbot, path: Path, root: Path):
    """Built, SHOWN, and loaded.

    Shown on purpose: a ``ViewBox`` with no geometry defers its range update,
    so an unshown pane reports the range it had before the zoom and every
    autoscaling assertion below would pass by accident.
    """
    from bidsmgr.gui.widgets.mrs_viewer_pane import MrsViewerPane

    pane = MrsViewerPane()
    qtbot.addWidget(pane)
    pane.resize(1000, 600)
    pane.show()
    qtbot.waitExposed(pane)
    pane.set_file(path, root)
    qtbot.waitUntil(lambda: pane._data is not None, timeout=30_000)
    return pane


def _curves(pane):
    return [item for item in pane._plot.getPlotItem().items
            if item.__class__.__name__ == "PlotDataItem"]


def _rows(pane):
    return [marker.label.orthoPos for marker in pane._markers]


class TestItOpens:
    def test_a_spectrum_is_read_and_drawn(self, qtbot, spectrum_file, tmp_path):
        pane = _pane(qtbot, spectrum_file, tmp_path)
        assert pane._data["nucleus"] == "1H"
        assert len(_curves(pane)) == 1

    def test_a_file_with_no_mrs_header_says_so(self, qtbot, tmp_path):
        """Rather than raising, because a NIfTI in an ``mrs/`` folder without
        the header is a real thing a converter can produce."""
        import nibabel as nib

        folder = tmp_path / "sub-001" / "mrs"
        folder.mkdir(parents=True)
        path = folder / "sub-001_svs.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4)), str(path))

        from bidsmgr.gui.widgets.mrs_viewer_pane import MrsViewerPane

        pane = MrsViewerPane()
        qtbot.addWidget(pane)
        pane.set_file(path, tmp_path)
        qtbot.waitUntil(lambda: pane._worker is None, timeout=30_000)
        assert pane._data is None
        assert "no NIfTI-MRS header" in pane._hint.text()


class TestTheMetaboliteLabels:
    def test_they_are_drawn_with_names(self, qtbot, spectrum_file, tmp_path):
        pane = _pane(qtbot, spectrum_file, tmp_path)
        assert len(pane._markers) > 1
        assert pane._marker_shifts == sorted(pane._marker_shifts)

    def test_crowded_names_are_moved_apart(self, qtbot, spectrum_file, tmp_path):
        """Creatine at 3.03 and choline at 3.22 are a fifth of a ppm apart.
        On one line, across the whole window, they overlap."""
        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._reset_view()
        assert len(set(_rows(pane))) > 1, "all on one row means they overlap"

    def test_zooming_in_gives_them_room_back(self, qtbot, spectrum_file, tmp_path):
        """A label's width in pixels does not change with the zoom, so the
        same pair that collided has the pane to itself close up."""
        from bidsmgr.gui.widgets.mrs_viewer_pane import _LABEL_ROWS

        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._plot.getPlotItem().setXRange(2.9, 3.4, padding=0)
        assert set(_rows(pane)) == {_LABEL_ROWS[0]}, "no need to stagger here"

    def test_turning_them_off_removes_them(self, qtbot, spectrum_file, tmp_path):
        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._show_metabolites.setChecked(False)
        assert pane._markers == []

    def test_a_row_is_never_off_the_bottom(self, qtbot, spectrum_file, tmp_path):
        from bidsmgr.gui.widgets.mrs_viewer_pane import _LABEL_ROWS

        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._plot.getPlotItem().setXRange(3.0, 3.05, padding=0)
        assert all(row in _LABEL_ROWS for row in _rows(pane))


class TestTheView:
    def test_panning_cannot_leave_the_data(self, qtbot, spectrum_file, tmp_path):
        """Dragging used to walk into an empty pane with no way back but
        Reset, which is the complaint that produced the limits."""
        pane = _pane(qtbot, spectrum_file, tmp_path)
        view = pane._plot.getPlotItem().getViewBox()
        pane._reset_view()
        width = np.ptp(view.viewRange()[0])
        for _ in range(60):
            view.translateBy(x=+2.0)
        low, high = view.state["limits"]["xLimits"]
        assert view.viewRange()[0][1] <= high + 1e-6
        for _ in range(120):
            view.translateBy(x=-2.0)
        assert view.viewRange()[0][0] >= low - 1e-6
        assert np.ptp(view.viewRange()[0]) == pytest.approx(width)

    def test_the_height_follows_the_visible_slice(
        self, qtbot, spectrum_file, tmp_path,
    ):
        """Zoomed past the water residual, the metabolites should fill the
        pane rather than sit flat against the axis."""
        pane = _pane(qtbot, spectrum_file, tmp_path)
        view = pane._plot.getPlotItem().getViewBox()
        assert view.state["autoVisibleOnly"][1] is True
        pane._clip.setChecked(False)
        qtbot.wait(50)          # pyqtgraph defers the auto-range update
        whole = np.ptp(view.viewRange()[1])
        pane._plot.getPlotItem().setXRange(3.3, 4.0, padding=0)
        qtbot.wait(50)
        quiet = np.ptp(view.viewRange()[1])
        assert quiet < whole, "a quiet stretch was scaled on the big peak"

    def test_fit_all_is_wider_than_the_standard_window(
        self, qtbot, spectrum_file, tmp_path,
    ):
        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._reset_view()
        standard = np.ptp(pane._plot.getPlotItem().getViewBox().viewRange()[0])
        pane._fit_all()
        everything = np.ptp(pane._plot.getPlotItem().getViewBox().viewRange()[0])
        assert everything > standard

    def test_the_standard_window_checkbox_actually_moves_the_view(
        self, qtbot, spectrum_file, tmp_path,
    ):
        """It used to recompute the span and leave the view where it was, so
        ticking it looked like it did nothing."""
        pane = _pane(qtbot, spectrum_file, tmp_path)
        view = pane._plot.getPlotItem().getViewBox()
        pane._clip.setChecked(True)
        clipped = np.ptp(view.viewRange()[0])
        pane._clip.setChecked(False)
        assert np.ptp(view.viewRange()[0]) > clipped

    def test_the_fid_page_draws_against_seconds(
        self, qtbot, spectrum_file, tmp_path,
    ):
        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._domain.setCurrentIndex(1)
        assert "Time" in pane._plot.getPlotItem().getAxis("bottom").labelText
        assert not pane._plot.getPlotItem().getViewBox().xInverted()
        pane._domain.setCurrentIndex(0)
        assert pane._plot.getPlotItem().getViewBox().xInverted(), (
            "chemical shift runs right to left"
        )


class TestTheSharedControls:
    def test_the_default_line_is_not_a_hairline(
        self, qtbot, spectrum_file, tmp_path,
    ):
        pane = _pane(qtbot, spectrum_file, tmp_path)
        assert pane._line_width >= 2
        assert _curves(pane)[0].opts["pen"].width() >= 2

    def test_the_line_popup_changes_the_trace(
        self, qtbot, spectrum_file, tmp_path,
    ):
        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._set_line_style(6, "#ff7b72")
        pen = _curves(pane)[0].opts["pen"]
        assert pen.width() == 6
        assert pen.color().name() == "#ff7b72"

    def test_one_trace_means_no_colour_by_type(self, qtbot, spectrum_file, tmp_path):
        """The choice only makes sense where there are types to tell apart."""
        from bidsmgr.gui.widgets.line_style_dialog import LineStyleDialog

        pane = _pane(qtbot, spectrum_file, tmp_path)
        dlg = LineStyleDialog(2, None, allow_by_type=False, parent=pane)
        qtbot.addWidget(dlg)
        assert not dlg._by_type.isEnabled()
        assert dlg._one_colour.isChecked()

    def test_the_crosshair_reads_out_ppm(self, qtbot, spectrum_file, tmp_path):
        from PyQt6.QtCore import QPointF

        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._cross["proxy"].sigDelayed.emit(
            (QPointF(pane._plot.sceneBoundingRect().center()),)
        )
        assert "ppm" in pane._readout.text()

    def test_the_crosshair_survives_a_redraw(self, qtbot, spectrum_file, tmp_path):
        """``clear()`` takes every item with it, the cross included."""
        pane = _pane(qtbot, spectrum_file, tmp_path)
        pane._broadening.setValue(5.0)
        items = pane._plot.getPlotItem().items
        assert pane._cross["v"] in items
        assert pane._cross["h"] in items


class TestTheme:
    def test_a_swap_while_open_repaints_the_plot(
        self, qtbot, spectrum_file, tmp_path,
    ):
        """pyqtgraph reads no QSS, so the palette has to be handed down. The
        historical failure was a plot that stayed dark until the app was
        restarted."""
        from bidsmgr.gui import theme_manager
        from bidsmgr.gui.theme_manager import DARK, LIGHT

        pane = _pane(qtbot, spectrum_file, tmp_path)
        before = pane._plot.backgroundBrush().color().name()
        original = theme_manager.CUR()
        other = LIGHT if original is DARK else DARK
        try:
            theme_manager._CURRENT = other
            pane.repaint_for_palette(other)
            assert pane._plot.backgroundBrush().color().name() != before
            from PyQt6.QtGui import QColor

            assert pane._cross["v"].pen.color() == QColor(other["muted"]), (
                "the crosshair follows the palette too"
            )
            assert pane._markers, "the labels survive a repaint"
        finally:
            theme_manager._CURRENT = original
            pane.repaint_for_palette(original)
