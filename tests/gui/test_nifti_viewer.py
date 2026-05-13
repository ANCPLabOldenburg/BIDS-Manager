"""Tests for the NIfTI 2-D slice viewer + file-kind routing.

The Editor's center pane routes ``.nii`` / ``.nii.gz`` to
:class:`NiftiViewerPane`. The pane loads via nibabel **on a worker
thread**, renders an axial / coronal / sagittal 2-D slice with a
crosshair, and reports voxel values on click.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.nifti_viewer_pane import (
    NiftiViewerPane,
    _AXIS_AXIAL,
    _AXIS_CORONAL,
    _AXIS_SAGITTAL,
    _load_nifti,
)


pytestmark = pytest.mark.gui


def _load_and_wait(pane, path, root, qtbot=None, timeout_ms: int = 5000):
    """Call :meth:`NiftiViewerPane.set_file` and block until done.

    The pane loads on a :class:`QThread` worker — tests need to wait
    for the ``loaded`` signal before inspecting ``_data`` /
    ``_grid_cells`` / etc. Falls back to blocking on the worker
    directly when no qtbot is supplied.
    """
    from PyQt6.QtWidgets import QApplication

    if qtbot is not None:
        with qtbot.waitSignal(pane.loaded, timeout=timeout_ms):
            pane.set_file(path, root)
        return
    pane.set_file(path, root)
    worker = pane._loader
    if worker is not None:
        worker.wait(timeout_ms)
    QApplication.processEvents()
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_nifti(path: Path, arr: np.ndarray) -> None:
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(img, str(path))


@pytest.fixture
def bids_root_with_nifti(tmp_path: Path) -> Path:
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    # 3-D test volume (10×12×8) with an obvious gradient so slice
    # rendering is non-trivial.
    arr_3d = np.arange(10 * 12 * 8, dtype=np.float32).reshape(10, 12, 8)
    _write_nifti(anat / "sub-01_ses-01_T1w.nii.gz", arr_3d)
    # A 4-D BOLD volume to exercise the volume slider.
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    arr_4d = np.random.default_rng(42).random(
        (6, 6, 4, 3), dtype=np.float32,
    )
    _write_nifti(
        func / "sub-01_ses-01_task-rest_bold.nii.gz", arr_4d,
    )
    # A plain ``.nii`` (no gz) to confirm routing matches both extensions.
    arr_plain = np.ones((4, 4, 4), dtype=np.float32)
    _write_nifti(anat / "sub-01_ses-01_T2w.nii", arr_plain)
    return root


# ---------------------------------------------------------------------------
# _load_nifti helper
# ---------------------------------------------------------------------------


def test_load_nifti_returns_array_and_meta(tmp_path: Path) -> None:
    p = tmp_path / "x.nii.gz"
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    _write_nifti(p, arr)
    img, data, meta = _load_nifti(p)
    assert data.shape == (2, 3, 4)
    assert not meta.get("is_rgb")
    assert img is not None


# ---------------------------------------------------------------------------
# NiftiViewerPane basics
# ---------------------------------------------------------------------------


def test_pane_starts_with_empty_hint(qapp, isolated_settings) -> None:
    pane = NiftiViewerPane()
    assert pane.current_file() is None
    assert pane._stack.currentIndex() == 0
    assert not pane._toolbar.isVisible()
    assert not pane._footer.isVisible()


def test_set_file_loads_nifti_and_shows_canvas(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    pane.show()  # needs to be visible for the visibility checks
    qapp.processEvents()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    assert pane.current_file() == t1
    assert pane._stack.currentWidget() is pane._canvas
    assert pane._toolbar.isVisible()
    assert pane._footer.isVisible()
    # Slice slider sized to the axial depth (8).
    assert pane._slice_slider.maximum() == 7
    # Volume slider disabled for 3-D data.
    assert pane._vol_slider.maximum() == 0
    assert not pane._vol_slider.isEnabled()
    # Crosshair defaults to the centre voxel.
    assert pane._cross_voxel == [5, 6, 4]


def test_set_file_handles_plain_nii(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    t2 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T2w.nii"
    )
    _load_and_wait(pane, t2, bids_root_with_nifti, qtbot=qtbot)
    assert pane.current_file() == t2
    assert pane._data is not None
    assert pane._data.shape == (4, 4, 4)


def test_4d_enables_volume_slider(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    assert pane._data is not None
    assert pane._data.shape == (6, 6, 4, 3)
    assert pane._vol_slider.maximum() == 2
    assert pane._vol_slider.isEnabled()


def test_set_file_none_clears(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    pane.set_file(None, None)
    qapp.processEvents()
    assert pane.current_file() is None
    assert pane._data is None
    assert pane._stack.currentIndex() == 0


def test_orientation_buttons_resize_slice_slider(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    # Axial (default): depth = shape[2] = 8.
    assert pane._slice_slider.maximum() == 7
    pane._sa_btn.click()
    qapp.processEvents()
    # Sagittal: depth = shape[0] = 10.
    assert pane._orientation == _AXIS_SAGITTAL
    assert pane._slice_slider.maximum() == 9
    pane._co_btn.click()
    qapp.processEvents()
    # Coronal: depth = shape[1] = 12.
    assert pane._orientation == _AXIS_CORONAL
    assert pane._slice_slider.maximum() == 11
    pane._ax_btn.click()
    qapp.processEvents()
    assert pane._orientation == _AXIS_AXIAL
    assert pane._slice_slider.maximum() == 7


def test_voxel_value_readout_reflects_data(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    # Centre voxel of the gradient: shape (10, 12, 8), centre (5,6,4).
    # data[5,6,4] = 5*12*8 + 6*8 + 4 = 480 + 48 + 4 = 532.
    assert "532" in pane._voxel_value.text()


def test_footer_summary_reports_shape(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    assert "10×12×8" in pane._footer_summary.text()
    # Path should be relative to root.
    assert "sub-01" in pane._footer_path.text()
    assert "T1w" in pane._footer_path.text()


def test_load_failure_surfaces_signal(
    qapp, qtbot, tmp_path: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bad = tmp_path / "not_a_nifti.nii.gz"
    bad.write_bytes(b"garbage bytes")
    with qtbot.waitSignal(pane.load_failed, timeout=2000):
        pane.set_file(bad, tmp_path)
    assert pane._data is None
    assert pane._stack.currentWidget() is pane._empty_hint
    # Hint mentions the file name.
    assert "not_a_nifti" in pane._empty_hint.text()


# ---------------------------------------------------------------------------
# Loading state (threaded)
# ---------------------------------------------------------------------------


def test_set_file_shows_loading_page_before_worker_completes(
    qapp, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    """The pane swaps to the loading page synchronously the moment
    ``set_file`` is called — the worker hasn't completed yet, but the
    user already sees the spinner."""
    pane = NiftiViewerPane()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    pane.set_file(t1, bids_root_with_nifti)
    # Worker hasn't been pumped through the event loop yet.
    assert pane._stack.currentWidget() is pane._loading_panel
    assert pane._loading_label.text().endswith(f"{t1.name}…")
    # Toolbar / footer stay hidden during load.
    assert not pane._toolbar.isVisible()
    assert not pane._footer.isVisible()
    # Now let the worker finish.
    pane._loader.wait(5000)
    qapp.processEvents()
    qapp.processEvents()
    assert pane._stack.currentWidget() is pane._canvas


def test_rapid_file_swap_discards_stale_load(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    """If the user picks a second file before the first finishes
    loading, the stale worker's result must be discarded."""
    pane = NiftiViewerPane()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    # Start t1 then immediately switch to bold.
    pane.set_file(t1, bids_root_with_nifti)
    pane.set_file(bold, bids_root_with_nifti)
    # Block until the second worker finishes.
    pane._loader.wait(5000)
    qapp.processEvents()
    qapp.processEvents()
    # Pane should reflect bold, not t1.
    assert pane.current_file() == bold
    assert pane._data is not None
    assert pane._data.shape == (6, 6, 4, 3)


# ---------------------------------------------------------------------------
# EditorPanel routing
# ---------------------------------------------------------------------------


def test_editor_routes_nii_to_nifti_viewer(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._set_root(bids_root_with_nifti, persist=False)
    qapp.processEvents()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    panel._on_file_selected(t1)
    qapp.processEvents()
    assert panel._center_stack.currentWidget() is panel._nifti_viewer
    assert panel._nifti_viewer.current_file() == t1


def test_editor_routes_nii_plain_to_nifti_viewer(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._set_root(bids_root_with_nifti, persist=False)
    t2 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T2w.nii"
    )
    panel._on_file_selected(t2)
    qapp.processEvents()
    assert panel._center_stack.currentWidget() is panel._nifti_viewer
    assert panel._nifti_viewer.current_file() == t2


# ---------------------------------------------------------------------------
# Tri-view multi-panel
# ---------------------------------------------------------------------------


def test_tri_view_toggle_swaps_image_stack(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    assert pane._image_stack.currentIndex() == 0
    assert pane._slice_slider.isEnabled()
    # Toggle on.
    pane._tri_btn.click()
    qapp.processEvents()
    assert pane._tri_view is True
    assert pane._image_stack.currentIndex() == 1
    # Orientation pills + slice slider greyed out in tri-view.
    assert not pane._sa_btn.isEnabled()
    assert not pane._co_btn.isEnabled()
    assert not pane._ax_btn.isEnabled()
    assert not pane._slice_slider.isEnabled()
    # All three labels rendered a pixmap.
    for axis in (0, 1, 2):
        pix = pane._tri_labels[axis].pixmap()
        assert pix is not None and not pix.isNull(), (
            f"tri-view axis {axis} has no pixmap"
        )
    # Toggle off restores single-pane.
    pane._tri_btn.click()
    qapp.processEvents()
    assert pane._tri_view is False
    assert pane._image_stack.currentIndex() == 0
    assert pane._sa_btn.isEnabled()
    assert pane._slice_slider.isEnabled()


def test_tri_view_click_propagates_crosshair_to_all_panels(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    """Clicking on the coronal panel should change i and k but leave
    j unchanged."""
    from PyQt6.QtCore import QPointF, Qt as QtMod
    from PyQt6.QtGui import QMouseEvent

    pane = NiftiViewerPane()
    pane.show()
    qapp.processEvents()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    pane._tri_btn.click()
    qapp.processEvents()
    pane.resize(900, 400)
    qapp.processEvents()

    coronal = pane._tri_labels[_AXIS_CORONAL]
    pix = coronal.pixmap()
    assert pix is not None and not pix.isNull()
    pw, ph = pix.width(), pix.height()
    lw, lh = coronal.width(), coronal.height()
    cx = (lw - pw) / 2 + pw // 2
    cy = (lh - ph) / 2 + ph // 2
    before = list(pane._cross_voxel)
    ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(cx, cy),
        QPointF(cx, cy),
        QtMod.MouseButton.LeftButton,
        QtMod.MouseButton.LeftButton,
        QtMod.KeyboardModifier.NoModifier,
    )
    pane._on_image_clicked(ev, _AXIS_CORONAL, coronal)
    qapp.processEvents()
    after = list(pane._cross_voxel)
    # j (coronal slice index) stays put.
    assert after[_AXIS_CORONAL] == before[_AXIS_CORONAL]
    # The whole crosshair is bounded by the data shape.
    for axis, dim in enumerate(pane._data.shape[:3]):
        assert 0 <= after[axis] < dim


def test_drag_moves_crosshair_continuously(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    """Mouse-move events (button held) should update the crosshair."""
    from PyQt6.QtCore import QPointF, Qt as QtMod
    from PyQt6.QtGui import QMouseEvent

    pane = NiftiViewerPane()
    pane.show()
    qapp.processEvents()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    pane.resize(700, 500)
    qapp.processEvents()

    label = pane._image_label
    pix = label.pixmap()
    assert pix is not None and not pix.isNull()
    pw, ph = pix.width(), pix.height()
    lw, lh = label.width(), label.height()
    off_x = (lw - pw) / 2
    off_y = (lh - ph) / 2

    # Two synthetic drag positions inside the pixmap.
    def _move_event(x, y, etype):
        return QMouseEvent(
            etype,
            QPointF(x, y),
            QPointF(x, y),
            QtMod.MouseButton.LeftButton,
            QtMod.MouseButton.LeftButton,
            QtMod.KeyboardModifier.NoModifier,
        )

    x1 = off_x + pw // 3
    y1 = off_y + ph // 3
    x2 = off_x + (pw * 2) // 3
    y2 = off_y + (ph * 2) // 3
    # Press at (x1, y1).
    label.mousePressEvent(_move_event(x1, y1, QMouseEvent.Type.MouseButtonPress))
    qapp.processEvents()
    cross_after_press = list(pane._cross_voxel)
    # Drag to (x2, y2).
    label.mouseMoveEvent(_move_event(x2, y2, QMouseEvent.Type.MouseMove))
    qapp.processEvents()
    cross_after_drag = list(pane._cross_voxel)
    # The crosshair moved between press and drag.
    assert cross_after_press != cross_after_drag


# ---------------------------------------------------------------------------
# Graph (4-D time-series) panel
# ---------------------------------------------------------------------------


def test_graph_button_disabled_for_3d(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    assert pane._graph_btn.isEnabled() is False


def test_graph_button_enabled_for_4d(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    assert pane._graph_btn.isEnabled() is True


def test_graph_toggle_shows_plot_with_voxel_timeseries(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    assert not pane._graph_panel.isVisible() or not pane._graph_visible
    pane._graph_btn.click()
    qapp.processEvents()
    assert pane._graph_visible is True
    # Default scope = 1 → one cell only (the centre voxel).
    assert len(pane._grid_cells) == 1
    cell = pane._grid_cells[0]
    assert cell["is_center"] is True
    x, y = cell["curve"].getData()
    assert len(x) == 3
    i, j, k = pane._cross_voxel
    expected = pane._data[i, j, k, :]
    assert np.allclose(y, expected)


def test_volume_slider_moves_graph_marker(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    pane._graph_btn.click()
    qapp.processEvents()
    pane._vol_slider.setValue(2)
    qapp.processEvents()
    marker = pane._grid_cells[0]["marker"]
    xs, _ys = marker.getData()
    assert int(xs[0]) == 2


def test_scope_spinbox_builds_neighbour_grid(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    """Scope 2 → 3×3 = 9 cells (when all in bounds); scope 3 → 5×5 = 25."""
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    pane._graph_btn.click()
    qapp.processEvents()
    pane._scope_spin.setValue(2)
    qapp.processEvents()
    assert len(pane._grid_cells) == 9
    pane._scope_spin.setValue(3)
    qapp.processEvents()
    assert len(pane._grid_cells) >= 9
    pane._scope_spin.setValue(1)
    qapp.processEvents()
    assert len(pane._grid_cells) == 1


def test_mark_neighbors_toggle_drops_outer_markers(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    pane._graph_btn.click()
    pane._scope_spin.setValue(2)
    qapp.processEvents()
    n_markers = sum(1 for c in pane._grid_cells if c["marker"] is not None)
    assert n_markers == len(pane._grid_cells)
    pane._mark_neighbors_box.setChecked(False)
    qapp.processEvents()
    centred = [c for c in pane._grid_cells if c["marker"] is not None]
    assert len(centred) == 1
    assert centred[0]["is_center"] is True


def test_dot_size_spinbox_changes_marker_size(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    pane._graph_btn.click()
    qapp.processEvents()
    pane._dot_size_spin.setValue(16)
    qapp.processEvents()
    marker = pane._grid_cells[0]["marker"]
    assert marker.opts["size"] == 16 or marker.points()[0].size() == 16


def test_plot_viewboxes_are_locked_against_mouse(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    pane._graph_btn.click()
    qapp.processEvents()
    for cell in pane._grid_cells:
        vb = cell["plot"].getViewBox()
        assert vb.state["mouseEnabled"] == [False, False]


def test_graph_toggle_off_hides_panel(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    pane._graph_btn.click()
    qapp.processEvents()
    assert pane._graph_visible is True
    pane._graph_btn.click()
    qapp.processEvents()
    assert pane._graph_visible is False
    assert not pane._graph_panel.isVisible()


def test_switching_from_4d_to_3d_closes_graph(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    bold = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.nii.gz"
    )
    _load_and_wait(pane, bold, bids_root_with_nifti, qtbot=qtbot)
    pane._graph_btn.click()
    qapp.processEvents()
    assert pane._graph_visible is True
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    _load_and_wait(pane, t1, bids_root_with_nifti, qtbot=qtbot)
    assert pane._graph_btn.isEnabled() is False
    assert pane._graph_btn.isChecked() is False
    assert pane._graph_visible is False


# ---------------------------------------------------------------------------
# Crosshair settings (persisted)
# ---------------------------------------------------------------------------


def test_crosshair_thickness_persists(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    """Changing the thickness spinbox writes to AppSettings and a new
    pane picks it up on construction."""
    from bidsmgr.gui.app_settings import AppSettings

    pane = NiftiViewerPane()
    pane._crosshair_thickness_spin.setValue(4)
    qapp.processEvents()
    assert pane._crosshair_thickness == 4
    assert AppSettings.load().nifti_crosshair_thickness == 4

    pane2 = NiftiViewerPane()
    assert pane2._crosshair_thickness == 4
    assert pane2._crosshair_thickness_spin.value() == 4


def test_crosshair_color_persists(
    qapp, isolated_settings,
) -> None:
    from PyQt6.QtGui import QColor
    from bidsmgr.gui.app_settings import AppSettings

    pane = NiftiViewerPane()
    # Skip the QColorDialog by writing the colour directly.
    pane._crosshair_color = QColor("#FF8800")
    pane._refresh_crosshair_swatch()
    pane._persist_crosshair()
    assert AppSettings.load().nifti_crosshair_color.lower() == "#ff8800"

    pane2 = NiftiViewerPane()
    assert pane2._crosshair_color.name().lower() == "#ff8800"


def test_editor_clears_nifti_viewer_when_switching_to_json(
    qapp, qtbot, bids_root_with_nifti: Path, isolated_settings,
) -> None:
    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._set_root(bids_root_with_nifti, persist=False)
    t1 = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.nii.gz"
    )
    json_path = (
        bids_root_with_nifti / "sub-01" / "ses-01" / "anat"
        / "sub-01_ses-01_T1w.json"
    )
    json_path.write_text('{"Manufacturer": "Siemens"}')

    panel._on_file_selected(t1)
    qapp.processEvents()
    assert panel._nifti_viewer.current_file() == t1

    panel._on_file_selected(json_path)
    qapp.processEvents()
    assert panel._center_stack.currentWidget() is panel._sidecar_form
    assert panel._nifti_viewer.current_file() is None
