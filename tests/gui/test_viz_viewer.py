"""The volume viewer, driven the way the Editor and a user drive it.

Replaces the tests of the old ``NiftiViewerPane`` (2-D slices, the
multi-planar view, the 4-D graph, the crosshair preferences, the remembered
layout, PET time axes). Everything here goes through the public surface:
``Viewer`` methods and signals, its actions (what a button, a key or a menu
entry does), its canvases, and the library's ``views`` queries, never a
private attribute of a widget.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.gui.viz.bridge import SettingsHub  # noqa: E402
from bidsmgr.viz import keynames, views  # noqa: E402

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _write(path: Path, arr: np.ndarray, affine=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, np.eye(4) if affine is None else affine), str(path))
    return path


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    # A gradient, so every voxel has its own value: data[i, j, k] = i*96 + j*8 + k.
    _write(anat / "sub-01_ses-01_T1w.nii.gz",
           np.arange(10 * 12 * 8, dtype=np.float32).reshape(10, 12, 8))
    _write(anat / "sub-01_ses-01_T2w.nii", np.ones((4, 4, 4), dtype=np.float32))
    _write(root / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-rest_bold.nii.gz",
           np.random.default_rng(42).random((6, 6, 4, 3), dtype=np.float32))
    return root


def _t1(root: Path) -> Path:
    return root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"


def _t2(root: Path) -> Path:
    return root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T2w.nii"


def _bold(root: Path) -> Path:
    return root / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-rest_bold.nii.gz"


@pytest.fixture
def no_gpu(monkeypatch):
    """The host's GPU must not decide what a 2-D test measures."""
    from bidsmgr.gui.viz.canvases import render

    monkeypatch.setattr(render, "gpu_available", lambda: False)


def _viewer(qtbot, *, show: bool = True, size=(1000, 640)) -> Viewer:
    viewer = Viewer(kind="volume")
    qtbot.addWidget(viewer)
    if show:
        viewer.resize(*size)
        viewer.show()
        qtbot.waitExposed(viewer)
    return viewer


def _open(qtbot, viewer: Viewer, path: Path, root=None, timeout=20_000) -> Viewer:
    with qtbot.waitSignal(viewer.loaded, timeout=timeout):
        viewer.set_file(path, root)
    viewer.qstore.flush()
    return viewer


def _settle(qtbot, viewer: Viewer) -> None:
    viewer.qstore.flush()
    qtbot.wait(5)
    for canvas in viewer.canvases("slice"):
        canvas.repaint()


def _click(qtbot, canvas, col: float, row: float, button=Qt.MouseButton.LeftButton) -> None:
    canvas.repaint()
    pt = canvas.grid_to_screen(col, row).toPoint()
    qtbot.mouseClick(canvas, button, pos=pt)


def _canvas(viewer: Viewer, plane: str):
    for c in viewer.canvases("slice"):
        if c.plane == plane:
            return c
    raise AssertionError(f"no {plane} canvas on screen")


# ---------------------------------------------------------------------------
# Opening and closing
# ---------------------------------------------------------------------------


def test_it_starts_with_a_hint(qtbot, no_gpu) -> None:
    viewer = _viewer(qtbot)
    assert viewer.current_file() is None
    assert viewer.page() == "hint"
    assert not viewer.toolbar_visible()


def test_a_volume_opens_on_its_centre(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.current_file() == _t1(ds)
    assert viewer.page() == "content" and viewer.toolbar_visible()
    assert views.cursor_voxel(viewer.store) == (5, 6, 4)
    # data[5, 6, 4] = 5*96 + 6*8 + 4 = 532, shown in the readout with its
    # millimetres and the FILE's own voxel indices.
    text = viewer.readout_text()
    assert "(5, 6, 4) = 532" in text and "mm" in text


def test_the_footer_names_the_file_and_its_shape(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.path_text() == "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
    assert "10x12x8" in viewer.summary_text()
    assert "float32" in viewer.summary_text()


def test_an_uncompressed_nifti_opens(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t2(ds), ds)
    assert viewer.source().spatial == (4, 4, 4)


def test_a_series_offers_its_volumes(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    src = viewer.source()
    assert src.n_frames == 3 and src.fully_loaded
    assert viewer.action("frame.next").isEnabled()
    viewer.trigger("frame.last")
    assert viewer.scene.base_layer().frame == 2


def test_a_single_volume_offers_no_volume_stepping(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert not viewer.action("frame.next").isEnabled()
    assert not viewer.action("view.graph").isEnabled()


def test_set_file_none_clears(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.set_file(None, None)
    assert viewer.current_file() is None
    assert viewer.source() is None
    assert viewer.page() == "hint"
    assert viewer.store.sources == {}


def test_a_bad_file_says_so(qtbot, tmp_path, no_gpu) -> None:
    viewer = _viewer(qtbot)
    bad = tmp_path / "not_a_nifti.nii.gz"
    bad.write_bytes(b"garbage bytes")
    with qtbot.waitSignal(viewer.load_failed, timeout=5000):
        viewer.set_file(bad, tmp_path)
    assert viewer.source() is None
    assert viewer.page() == "hint"
    assert "not_a_nifti" in viewer.hint_text()


def test_the_loading_page_shows_at_once(qtbot, ds, no_gpu) -> None:
    """The spinner appears the moment the file is chosen, before any read."""
    viewer = _viewer(qtbot)
    with qtbot.waitSignal(viewer.loaded, timeout=20_000):
        viewer.set_file(_t1(ds), ds)
        assert viewer.page() == "loading"
        assert not viewer.toolbar_visible()
    assert viewer.page() == "content"


def test_a_quick_second_choice_wins(qtbot, ds, no_gpu) -> None:
    """A stale read for the first file must not land on the second."""
    viewer = _viewer(qtbot)
    with qtbot.waitSignal(viewer.loaded, timeout=20_000) as blocker:
        viewer.set_file(_t1(ds), ds)
        viewer.set_file(_bold(ds), ds)
    assert blocker.args == [_bold(ds)]
    qtbot.wait(100)
    assert viewer.current_file() == _bold(ds)
    assert viewer.source().n_frames == 3


def test_reopening_the_same_file_twice_is_clean(qtbot, ds, no_gpu) -> None:
    """The generation number, not the path, decides staleness."""
    viewer = _viewer(qtbot)
    viewer.set_file(_t1(ds), ds)
    _open(qtbot, viewer, _t1(ds), ds)
    qtbot.wait(100)
    assert viewer.source().fully_loaded
    assert views.cursor_voxel(viewer.store) == (5, 6, 4)


# ---------------------------------------------------------------------------
# The Editor routes volumes here
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", [_t1, _t2])
def test_the_editor_shows_a_volume_in_the_viewer(qtbot, ds, no_gpu, which) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._set_root(ds, persist=False)
    viewer = panel._nifti_viewer
    with qtbot.waitSignal(viewer.loaded, timeout=20_000):
        panel._on_file_selected(which(ds))
    assert panel._center_stack.currentWidget() is viewer
    assert viewer.current_file() == which(ds)


def test_the_editor_clears_the_viewer_for_a_sidecar(qtbot, ds, no_gpu) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._set_root(ds, persist=False)
    sidecar = _t1(ds).with_name("sub-01_ses-01_T1w.json")
    sidecar.write_text('{"Manufacturer": "Siemens"}')
    with qtbot.waitSignal(panel._nifti_viewer.loaded, timeout=20_000):
        panel._on_file_selected(_t1(ds))
    panel._on_file_selected(sidecar)
    assert panel._center_stack.currentWidget() is panel._sidecar_form
    assert panel._nifti_viewer.current_file() is None


# ---------------------------------------------------------------------------
# One plane
# ---------------------------------------------------------------------------


def test_the_plane_buttons_size_the_slice_slider(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    for action, plane, depth in (("view.axial", "axial", 8),
                                 ("view.sagittal", "sagittal", 10),
                                 ("view.coronal", "coronal", 12)):
        viewer.action(action).trigger()           # from the Layout menu
        viewer.qstore.flush()
        assert viewer.scene.mode == "single" and viewer.scene.plane == plane
        assert viewer.presenter.layout_button.text() == plane.capitalize()
        assert views.slice_count(viewer.store, plane) == depth
        assert viewer.presenter.slice_control.maximum() == depth - 1


def test_the_slice_slider_moves_the_crosshair(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.axial")
    viewer.qstore.flush()
    viewer.presenter.slice_control.type_value(1)
    assert views.cursor_voxel(viewer.store)[2] == 1


def test_a_click_moves_the_crosshair_to_the_voxel_under_it(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.axial")
    _settle(qtbot, viewer)
    canvas = _canvas(viewer, "axial")
    grid = views.grid(viewer.store, "axial")
    # Axial: columns run toward +x (i), rows down from +y (j).
    want_i, want_j = 2, 9
    col = want_i
    row = grid.shape[0] - 1 - want_j
    _click(qtbot, canvas, col, row)
    assert views.cursor_voxel(viewer.store) == (want_i, want_j, 4)
    viewer.qstore.flush()   # notifications are coalesced to one per event-loop turn
    assert f"({want_i}, {want_j}, 4)" in viewer.readout_text()


def test_dragging_moves_the_crosshair_continuously(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.axial")
    _settle(qtbot, viewer)
    canvas = _canvas(viewer, "axial")
    a = canvas.grid_to_screen(1, 1).toPoint()
    b = canvas.grid_to_screen(7, 9).toPoint()
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=a)
    first = views.cursor_voxel(viewer.store)
    qtbot.mouseMove(canvas, pos=b)
    # QTest.mouseMove carries no button state on every platform; send the
    # move a held button produces.
    from PyQt6.QtGui import QMouseEvent

    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(b), QPointF(b),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    ))
    second = views.cursor_voxel(viewer.store)
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=b)
    assert first != second


def test_the_wheel_steps_slices(qtbot, ds, no_gpu) -> None:
    from PyQt6.QtGui import QWheelEvent

    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.axial")
    _settle(qtbot, viewer)
    canvas = _canvas(viewer, "axial")
    before = views.slice_index(viewer.store, "axial")
    ev = QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(0, 0), QPoint(0, -120),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                     Qt.ScrollPhase.NoScrollPhase, False)
    canvas.wheelEvent(ev)
    assert views.slice_index(viewer.store, "axial") == before + 1


def test_a_horizontal_scroll_steps_volumes(qtbot, ds, no_gpu) -> None:
    """Shift+wheel arrives as a horizontal scroll on X11 and macOS."""
    from PyQt6.QtGui import QWheelEvent

    viewer = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    viewer.trigger("view.axial")
    _settle(qtbot, viewer)
    canvas = _canvas(viewer, "axial")
    ev = QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(0, 0), QPoint(120, 0),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                     Qt.ScrollPhase.NoScrollPhase, False)
    canvas.wheelEvent(ev)
    assert viewer.scene.base_layer().frame == 1


# ---------------------------------------------------------------------------
# Three planes
# ---------------------------------------------------------------------------


def test_multi_planar_shows_three_planes(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("view.mode", mode="single")
    viewer.action("view.multi").trigger()
    _settle(qtbot, viewer)
    assert viewer.scene.mode == "multi"
    assert sorted(c.plane for c in viewer.canvases("slice")) == ["axial", "coronal", "sagittal"]
    for canvas in viewer.canvases("slice"):
        image = canvas.grab_image()
        assert not image.isNull()
    # Toggling again returns to one plane.
    viewer.action("view.multi").trigger()
    viewer.qstore.flush()
    assert viewer.scene.mode == "single"


def test_a_click_on_the_coronal_plane_keeps_its_slice(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("view.mode", mode="multi")
    _settle(qtbot, viewer)
    before = views.cursor_voxel(viewer.store)
    canvas = _canvas(viewer, "coronal")
    _click(qtbot, canvas, 1, 1)
    after = views.cursor_voxel(viewer.store)
    assert after[1] == before[1]            # j is the coronal slice
    assert after != before
    for axis, dim in enumerate((10, 12, 8)):
        assert 0 <= after[axis] < dim


def _thick(root: Path) -> Path:
    """A thick-slice scan: 1 x 1 mm in plane, 5 mm between slices."""
    return _write(root / "sub-01" / "anat" / "sub-01_FLAIR.nii.gz",
                  np.arange(40 * 40 * 8, dtype=np.float32).reshape(40, 40, 8),
                  affine=np.diag([1.0, 1.0, 5.0, 1.0]))


def test_thick_slices_are_drawn_to_scale(qtbot, tmp_path, no_gpu) -> None:
    """Fitted by millimetres, not voxels: a 5 mm slice is five times taller."""
    viewer = _open(qtbot, _viewer(qtbot, size=(1200, 520)), _thick(tmp_path), tmp_path)
    viewer.run("view.mode", mode="multi")
    _settle(qtbot, viewer)
    for canvas in viewer.canvases("slice"):
        grid = views.grid(viewer.store, canvas.plane)
        rows, cols = grid.shape
        a = canvas.grid_to_screen(0, 0)
        b = canvas.grid_to_screen(cols - 1, rows - 1)
        px_per_mm_across = (b.x() - a.x()) / ((cols - 1) * grid.pixel_mm[0])
        px_per_mm_down = (b.y() - a.y()) / ((rows - 1) * grid.pixel_mm[1])
        assert px_per_mm_down == pytest.approx(px_per_mm_across, rel=0.03), canvas.plane
    coronal = _canvas(viewer, "coronal")
    grid = views.grid(viewer.store, "coronal")
    a, b = coronal.grid_to_screen(0, 0), coronal.grid_to_screen(1, 1)
    assert (b.y() - a.y()) / (b.x() - a.x()) == pytest.approx(5.0, rel=0.03)


def test_a_click_on_a_thick_slice_lands_on_its_voxel(qtbot, tmp_path, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot, size=(1200, 520)), _thick(tmp_path), tmp_path)
    viewer.run("view.mode", mode="multi")
    _settle(qtbot, viewer)
    coronal = _canvas(viewer, "coronal")
    grid = views.grid(viewer.store, "coronal")
    j = views.cursor_voxel(viewer.store)[1]
    want = (30, j, 6)
    # Coronal: columns toward +x (i), rows down from +z (k).
    _click(qtbot, coronal, want[0], grid.shape[0] - 1 - want[2])
    assert views.cursor_voxel(viewer.store) == want


def _letters_clear_of_the_image(viewer: Viewer) -> int:
    """Every orientation letter of every slice on screen lies outside the
    part of the image that is drawn, and inside the widget. Returns how
    many letters were checked."""
    checked = 0
    for canvas in viewer.canvases("slice"):
        canvas.repaint()
        drawn = canvas._image_rect.intersected(canvas._image_area())
        assert not drawn.isEmpty(), canvas.plane
        boxes = canvas.letter_boxes()
        if not canvas.letters_shown():     # a thumbnail: none at all
            assert boxes == {}, canvas.plane
            continue
        assert set(boxes) == {"left", "right", "top", "bottom"}, canvas.plane
        inside = canvas.rect().toRectF()
        for side, box in boxes.items():
            assert not box.intersects(drawn), (canvas.plane, side, box, drawn)
            assert inside.contains(box), (canvas.plane, side, box, inside)
            checked += 1
    return checked


@pytest.mark.parametrize("size", [(1000, 640), (1400, 360), (720, 900), (600, 560)])
def test_the_orientation_letters_are_never_on_the_image(qtbot, ds, no_gpu, size) -> None:
    """Wide, tall and tiny, one plane and three, with the colour bar and
    the captions: the letters sit beside the image, never on it."""
    viewer = _open(qtbot, _viewer(qtbot, size=size), _t1(ds), ds)
    viewer.presenter.set_inspector(False, remember=False)
    viewer.run("view.flag", flag="labels", value=True)
    viewer.run("view.flag", flag="colorbar", value=True)
    for mode in ("hero", "multi"):
        viewer.run("view.mode", mode=mode)
        _settle(qtbot, viewer)
        qtbot.waitUntil(lambda: any(c.letters_shown() for c in viewer.canvases("slice")))
        assert _letters_clear_of_the_image(viewer) >= 4


def test_a_zoomed_image_stops_short_of_the_letters(qtbot, ds, no_gpu) -> None:
    """Zoomed past the canvas and panned, the image is cut at its area and
    the letters keep their band."""
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("view.flag", flag="labels", value=True)
    viewer.run("view.mode", mode="multi")
    for plane in ("axial", "coronal", "sagittal"):
        viewer.run("view.zoom", plane=plane, factor=6.0)
        viewer.run("view.pan", plane=plane, dx=1.5, dy=-1.0)
    _settle(qtbot, viewer)
    for canvas in viewer.canvases("slice"):
        area = canvas._image_area()
        assert canvas._image_rect.width() > area.width(), canvas.plane
    assert _letters_clear_of_the_image(viewer) == 12


def test_a_thumbnail_slice_drops_its_letters(qtbot, ds, no_gpu) -> None:
    """Too small for the letters' bands to leave a usable image: no letters,
    and the image keeps the room."""
    viewer = _open(qtbot, _viewer(qtbot, size=(1000, 640)), _t1(ds), ds)
    viewer.run("view.flag", flag="labels", value=True)
    viewer.run("view.mode", mode="multi")
    _settle(qtbot, viewer)
    canvas = viewer.canvases("slice")[0]
    assert canvas.letters_shown()
    canvas.setFixedSize(60, 60)
    canvas.repaint()
    assert not canvas.letters_shown()
    assert canvas.letter_boxes() == {}
    assert canvas._image_area().width() >= 50


def test_without_letters_the_image_takes_their_room(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot, size=(1000, 300)), _t1(ds), ds)
    viewer.run("view.mode", mode="hero")
    viewer.run("view.flag", flag="labels", value=True)
    _settle(qtbot, viewer)
    canvas = viewer.canvases("slice")[0]
    with_letters = canvas._image_rect.height()
    viewer.run("view.flag", flag="labels", value=False)
    _settle(qtbot, viewer)
    assert canvas._image_rect.height() > with_letters
    assert canvas.letter_boxes() == {}


# ---------------------------------------------------------------------------
# The time-course graph
# ---------------------------------------------------------------------------


def _with_graph(qtbot, ds, path=None, root=None):
    viewer = _open(qtbot, _viewer(qtbot), path or _bold(ds), root or ds)
    viewer.run("view.graph", value=True)
    viewer.qstore.flush()
    qtbot.waitUntil(lambda: bool(viewer.canvases("graph")), timeout=5000)
    graph = viewer.canvases("graph")[0]
    graph.refresh()
    return viewer, graph


def test_the_graph_plots_the_voxel_under_the_crosshair(qtbot, ds, no_gpu) -> None:
    viewer, graph = _with_graph(qtbot, ds)
    assert viewer.scene.graph_visible
    i, j, k = views.cursor_voxel(viewer.store)
    expected = nib.load(str(_bold(ds))).get_fdata()[i, j, k, :]
    assert np.allclose(graph.center_series(), expected)


def test_the_graph_follows_the_crosshair(qtbot, ds, no_gpu) -> None:
    viewer, graph = _with_graph(qtbot, ds)
    viewer.run("cursor.set_voxel", i=1, j=2, k=3)
    viewer.qstore.flush()
    expected = nib.load(str(_bold(ds))).get_fdata()[1, 2, 3, :]
    assert np.allclose(graph.center_series(), expected)


def test_the_graph_marker_follows_the_volume(qtbot, ds, no_gpu) -> None:
    viewer, graph = _with_graph(qtbot, ds)
    viewer.presenter.frame_control.type_value(2)
    viewer.qstore.flush()
    x, y = graph.marker_points()[0]
    assert x == 2.0 and y == pytest.approx(graph.center_series()[2])


def test_scope_builds_a_neighbour_grid(qtbot, ds, no_gpu) -> None:
    viewer, graph = _with_graph(qtbot, ds)
    graph.scope_combo.setCurrentIndex(graph.scope_combo.findData(2))
    viewer.qstore.flush()
    assert graph.cell_count() == 9
    assert len(graph.marker_points()) == 9
    graph.marks_action.setChecked(False)
    viewer.qstore.flush()
    assert len(graph.marker_points()) == 1
    graph._marker_actions[16].trigger()
    viewer.qstore.flush()
    assert graph.marker_size() == 16
    graph.scope_combo.setCurrentIndex(graph.scope_combo.findData(1))
    viewer.qstore.flush()
    assert graph.cell_count() == 1


def test_a_neighbourhood_at_the_edge_skips_what_is_outside(qtbot, ds, no_gpu) -> None:
    viewer, graph = _with_graph(qtbot, ds)
    viewer.run("cursor.set_voxel", i=0, j=0, k=0)
    graph.scope_combo.setCurrentIndex(graph.scope_combo.findData(2))
    viewer.qstore.flush()
    assert graph.cell_count() == 4      # a corner keeps 2 x 2 of its 3 x 3


def test_the_graph_cannot_be_dragged_away(qtbot, ds, no_gpu) -> None:
    _viewer_, graph = _with_graph(qtbot, ds)
    assert graph.mouse_locked()


def test_percent_change_scaling(qtbot, ds, no_gpu) -> None:
    viewer, graph = _with_graph(qtbot, ds)
    raw = graph.center_series().copy()
    graph.scaling_combo.setCurrentIndex(graph.scaling_combo.findData("percent"))
    viewer.qstore.flush()
    assert viewer.scene.graph.scaling == "percent"
    expected = (raw - raw.mean()) / raw.mean() * 100.0
    assert np.allclose(graph.center_series(), expected)


@pytest.mark.parametrize("scope", [1, 2])
def test_clicking_the_graph_jumps_to_that_volume(qtbot, ds, no_gpu, scope) -> None:
    viewer, graph = _with_graph(qtbot, ds)
    viewer.run("graph.set", scope=scope)
    viewer.run("frame.set", frame=0)
    viewer.qstore.flush()
    pos = graph.view_to_scene(2.0, float(np.mean(graph.center_series())))

    class _Click:
        def scenePos(self):
            return pos

        def double(self):
            return False

        def accept(self):
            pass

    graph._on_click(_Click())
    assert viewer.scene.base_layer().frame == 2


def test_the_graph_closes_for_a_single_volume(qtbot, ds, no_gpu) -> None:
    viewer, _graph = _with_graph(qtbot, ds)
    _open(qtbot, viewer, _t1(ds), ds)
    assert not viewer.action("view.graph").isEnabled()
    assert viewer.canvases("graph") == []
    viewer.button("view.graph").click()   # disabled: does nothing
    assert viewer.canvases("graph") == []


def test_the_graph_toggle_hides_it(qtbot, ds, no_gpu) -> None:
    viewer, _graph = _with_graph(qtbot, ds)
    viewer.button("view.graph").click()
    viewer.qstore.flush()
    assert not viewer.scene.graph_visible
    assert viewer.canvases("graph") == []


# -- PET: a time axis in real seconds -----------------------------------

DURATIONS = [10] * 6 + [30] * 4 + [60] * 5 + [300] * 5


def _starts() -> list[float]:
    out, t = [], 0.0
    for d in DURATIONS:
        out.append(t)
        t += d
    return out


def _pet(tmp_path: Path, *, name="sub-001_pet", sidecar=None) -> Path:
    d = tmp_path / "sub-001" / ("pet" if name.endswith("_pet") else "func")
    n = len(DURATIONS)
    mid = np.asarray(_starts()) + np.asarray(DURATIONS) / 2
    data = np.zeros((6, 6, 3, n), dtype=np.float32)
    data[:] = (40 * np.exp(-mid / 200) + 12 * (1 - np.exp(-mid / 60)))[None, None, None, :]
    img = _write(d / f"{name}.nii.gz", data)
    payload = {"FrameTimesStart": _starts(), "FrameDuration": DURATIONS}
    (d / f"{name}.json").write_text(json.dumps(payload if sidecar is None else sidecar))
    return img


def test_pet_is_graphed_against_mid_frame_seconds(qtbot, tmp_path, no_gpu) -> None:
    """Uneven frames: 10 s early, 300 s late. An index axis flattens the
    uptake, exactly where the kinetics are."""
    viewer, graph = _with_graph(qtbot, None, _pet(tmp_path), tmp_path)
    x = graph.x_values()
    mids = np.asarray(_starts()) + np.asarray(DURATIONS) / 2
    assert graph.x_is_time()
    assert np.allclose(x, mids)
    assert x[1] - x[0] == pytest.approx(10.0)
    assert x[-1] - x[-2] == pytest.approx(300.0)
    # The marker lands on the frame's real time, not its index.
    viewer.trigger("frame.last")
    viewer.qstore.flush()
    assert graph.marker_points()[0][0] == pytest.approx(mids[-1])


def test_a_series_without_timing_keeps_the_volume_axis(qtbot, tmp_path, no_gpu) -> None:
    """A BOLD file carrying a stray PET key keeps the volume number."""
    img = _pet(tmp_path, name="sub-001_task-rest_bold")
    _viewer_, graph = _with_graph(qtbot, None, img, tmp_path)
    assert not graph.x_is_time()
    assert list(graph.x_values()) == list(range(len(DURATIONS)))


def test_the_time_axis_can_be_forced_to_volumes(qtbot, tmp_path, no_gpu) -> None:
    viewer, graph = _with_graph(qtbot, None, _pet(tmp_path), tmp_path)
    graph.x_combo.setCurrentIndex(graph.x_combo.findData("frames"))
    viewer.qstore.flush()
    assert not graph.x_is_time()


@pytest.mark.parametrize("sidecar", [
    {"FrameTimesStart": [0], "FrameDuration": [300]},
    {"FrameTimesStart": ["early", "late"]},
])
def test_unusable_pet_timing_falls_back_to_volumes(qtbot, tmp_path, no_gpu, sidecar) -> None:
    _viewer_, graph = _with_graph(qtbot, None, _pet(tmp_path, sidecar=sidecar), tmp_path)
    assert not graph.x_is_time()


# ---------------------------------------------------------------------------
# Crosshair preferences
# ---------------------------------------------------------------------------


def test_crosshair_thickness_is_remembered(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    _column(viewer).view.cross_width.type_value(4)
    assert SettingsHub.instance().settings.crosshair.thickness == 4
    # A new session reads it back from disk.
    SettingsHub.reset_instance()
    again = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert _column(again).view.cross_width.value() == 4


def test_crosshair_colour_is_remembered(qtbot, ds, no_gpu) -> None:
    SettingsHub.instance().update(lambda s: setattr(s.crosshair, "color", "#ff8800"))
    SettingsHub.reset_instance()
    assert SettingsHub.instance().settings.crosshair.color.lower() == "#ff8800"


def test_a_preference_reaches_an_open_viewer_at_once(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    SettingsHub.instance().update(lambda s: setattr(s.crosshair, "thickness", 3))
    assert _column(viewer).view.cross_width.value() == 3


# ---------------------------------------------------------------------------
# The layout a volume opens in
# ---------------------------------------------------------------------------


def _remember(mode: str = "", plane: str = "axial") -> None:
    def keep(s):
        s.volume.mode = mode
        s.volume.plane = plane
    SettingsHub.instance().update(keep)
    SettingsHub.reset_instance()


def test_the_first_volume_opens_multi_planar_without_a_gpu(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.scene.mode == "multi"
    assert viewer.scene.plane == "axial"


@pytest.mark.parametrize("mode", ["single", "multi", "hero", "mosaic"])
def test_the_remembered_layout_is_what_opens(qtbot, ds, no_gpu, mode) -> None:
    _remember(mode)
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.scene.mode == mode


@pytest.mark.parametrize("mode", ["3d", "combo"])
def test_a_remembered_3d_layout_falls_back_to_the_planes(qtbot, ds, no_gpu, mode) -> None:
    """It used to land on a single axial slice."""
    _remember(mode)
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.scene.mode == "multi"


def test_a_junk_setting_is_ignored(qtbot, ds, no_gpu) -> None:
    from bidsmgr.gui.app_settings import AppSettings, KEYS

    AppSettings._settings().setValue(KEYS["viz_settings"], json.dumps(
        {"volume": {"mode": "sideways", "plane": "diagonal"}}))
    SettingsHub.reset_instance()
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.scene.mode == "multi" and viewer.scene.plane == "axial"


def test_switching_layout_is_remembered_at_once(qtbot, ds, no_gpu) -> None:
    """Written as the user switches: the Editor is not always closed cleanly."""
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.action("view.sagittal").trigger()
    viewer.qstore.flush()
    s = SettingsHub.instance().settings.volume
    assert (s.mode, s.plane) == ("single", "sagittal")
    viewer.action("view.multi").trigger()
    viewer.qstore.flush()
    assert SettingsHub.instance().settings.volume.mode == "multi"


def test_the_next_file_of_the_same_kind_keeps_the_current_view(qtbot, ds, no_gpu) -> None:
    """Only a new KIND of file changes the arrangement; then you are driving."""
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.coronal")
    viewer.set_file(None, None)          # the Editor shows a sidecar in between
    _open(qtbot, viewer, _t2(ds), ds)
    assert (viewer.scene.mode, viewer.scene.plane) == ("single", "coronal")


def test_a_cleared_viewer_keeps_a_3d_arrangement(qtbot, ds, no_gpu) -> None:
    """User report, 2026-10-09: a T1 in 3-D, a click on a JSON, the T1 again,
    and it was in three planes. With nothing open, no image is "3-D
    capable", so the arrangement was rewritten to "multi" while the viewer
    was empty, and the next image of the same kind inherited it (and the
    next change saved it, so 3-D was lost for good)."""
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.set_file(None, None)          # the Editor shows a sidecar
    viewer.ctx.scene.mode = "3d"         # the arrangement kept from a 3-D view
    viewer.presenter.apply_mode()
    assert viewer.scene.mode == "3d", "the empty viewer rewrote the arrangement"


def test_a_change_still_settling_is_kept_when_the_viewer_is_cleared(qtbot, ds, no_gpu) -> None:
    """A 3-D tweak is written once it settles; a click on a JSON in that
    moment used to drop it."""
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("clip.toggle")
    viewer.qstore.flush()
    assert viewer.presenter._view_timer.isActive()
    viewer.set_file(None, None)
    clips = SettingsHub.instance().settings.volume_look.get("clips") or []
    assert clips and clips[0]["active"], "the clip plane was not remembered"


def test_each_kind_of_file_opens_in_its_own_layout(qtbot, ds, no_gpu) -> None:
    """A BOLD run opens with its graph and a T1 without."""
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.presenter.layout_id == "mri.anat" and not viewer.scene.graph_visible
    _open(qtbot, viewer, _bold(ds), ds)
    assert viewer.presenter.layout_id == "mri.func"
    assert viewer.scene.graph_visible and viewer.scene.mode == "multi"
    _open(qtbot, viewer, _t1(ds), ds)
    assert not viewer.scene.graph_visible


def test_each_kind_remembers_the_layout_you_gave_it(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    viewer.run("view.graph", value=False)
    viewer.trigger("view.sagittal")
    viewer.qstore.flush()
    _open(qtbot, viewer, _t1(ds), ds)
    viewer.run("view.mode", mode="hero")
    viewer.qstore.flush()
    _open(qtbot, viewer, _bold(ds), ds)
    assert (viewer.scene.mode, viewer.scene.plane, viewer.scene.graph_visible) == \
        ("single", "sagittal", False)
    _open(qtbot, viewer, _t1(ds), ds)
    assert viewer.scene.mode == "hero"
    # And on disk, per kind.
    state = SettingsHub.instance().settings.layout_state
    assert state["mri.func"]["mode"] == "single" and state["mri.anat"]["mode"] == "hero"


def test_the_graph_height_is_remembered(qtbot, ds, no_gpu) -> None:
    viewer, _graph = _with_graph(qtbot, ds)
    split = viewer.presenter.vsplit
    total = sum(split.sizes())
    split.setSizes([int(total * 0.5), total - int(total * 0.5)])
    viewer.presenter.remember_sizes("volume.graph", split)
    qtbot.waitUntil(lambda: "volume.graph" in SettingsHub.instance().settings.layout_sizes,
                    timeout=3000)
    fresh = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    fresh.qstore.flush()
    qtbot.waitUntil(lambda: bool(fresh.canvases("graph")), timeout=5000)
    a, b = fresh.presenter.vsplit.sizes()
    assert a / (a + b) == pytest.approx(0.5, abs=0.05)


def test_a_saved_view_applies_to_another_image(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("view.mode", mode="hero")
    viewer.run("view.plane", plane="coronal")
    viewer.run("view.flag", flag="colorbar", value=True)
    viewer.presenter.save_view("Coronal hero")
    assert "Coronal hero" in SettingsHub.instance().settings.view_presets
    viewer.run("view.mode", mode="multi")
    viewer.run("view.flag", flag="colorbar", value=False)
    _open(qtbot, viewer, _bold(ds), ds)
    assert viewer.presenter.apply_view("Coronal hero")
    assert (viewer.scene.mode, viewer.scene.plane, viewer.scene.display.colorbar) == \
        ("hero", "coronal", True)
    # The view is an arrangement and a look, never a position.
    assert "cursor" not in SettingsHub.instance().settings.view_presets["Coronal hero"]
    viewer.presenter.delete_view("Coronal hero")
    assert "Coronal hero" not in SettingsHub.instance().settings.view_presets


def test_a_preset_gives_an_image_only_what_it_has(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    viewer.run("view.mode", mode="hero")
    viewer.run("view.graph", value=True)
    viewer.run("graph.set", qc_rows=["fd", "carpet"])
    viewer.trigger("graph.beside")
    viewer.qstore.flush()
    viewer.presenter.save_view("Run check")
    # On an image with no time series: the layout, never the time course.
    _open(qtbot, viewer, _t1(ds), ds)
    viewer.run("view.mode", mode="multi")
    viewer.run("view.graph", value=False)
    placement, rows = viewer.scene.layout.graph, list(viewer.scene.graph.qc_rows)
    assert viewer.presenter.apply_view("Run check")
    assert viewer.scene.mode == "hero"
    assert not viewer.scene.graph_visible
    assert not viewer.action("view.graph").isChecked(), "not checked where it cannot show"
    assert (viewer.scene.layout.graph, viewer.scene.graph.qc_rows) == (placement, rows), \
        "nothing of the time course is applied"
    # On another series: all of it.
    _open(qtbot, viewer, _bold(ds), ds)
    viewer.run("view.graph", value=False)
    viewer.run("graph.set", qc_rows=["dvars"])
    assert viewer.presenter.apply_view("Run check")
    assert viewer.scene.graph_visible and viewer.scene.graph.qc_rows == ["fd", "carpet"]
    assert viewer.scene.layout.graph == "right"


def test_a_preset_saved_without_a_series_leaves_a_series_alone(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("view.mode", mode="mosaic")
    viewer.presenter.save_view("Mosaic")
    assert "graph_visible" not in SettingsHub.instance().settings.view_presets["Mosaic"]
    _open(qtbot, viewer, _bold(ds), ds)
    viewer.run("view.graph", value=True)
    viewer.run("graph.set", qc_rows=["dvars"])
    assert viewer.presenter.apply_view("Mosaic")
    assert viewer.scene.mode == "mosaic"
    assert viewer.scene.graph_visible and viewer.scene.graph.qc_rows == ["dvars"]


def test_the_views_menu_lists_the_saved_views(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.presenter.save_view("Mine")
    menu = viewer.presenter.save_button.menu()
    viewer.presenter._fill_save_menu(menu)
    entries = {a.text(): a for a in menu.actions()}
    assert "Save the look as a preset..." in entries
    assert [a.text() for a in entries["Apply a preset"].menu().actions()] == ["Mine"]


def test_a_second_viewer_opens_the_way_the_first_was_left(qtbot, ds, no_gpu) -> None:
    first = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    first.trigger("view.coronal")
    first.qstore.flush()
    SettingsHub.reset_instance()        # a new Editor session, in effect
    second = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert (second.scene.mode, second.scene.plane) == ("single", "coronal")


def test_display_options_survive_the_next_file(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.radiological")
    viewer.trigger("view.colorbar")
    _open(qtbot, viewer, _bold(ds), ds)
    assert viewer.scene.display.radiological and viewer.scene.display.colorbar


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------


def test_the_window_is_in_data_units(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    rc = _column(viewer).control("window").range
    src = viewer.source()
    exact = viewer.scene.base_layer().display.window
    assert exact == pytest.approx(src.robust_range(0))
    # The fields show it to a sensible number of decimals for its size...
    assert (rc.lo_spin.value(), rc.hi_spin.value()) == pytest.approx(exact, abs=0.05)
    assert rc.hi_spin.decimals() == 1
    # ...and editing one end leaves the other EXACTLY as it was.
    rc.hi_spin.setValue(400.0)
    assert viewer.scene.base_layer().display.window == (exact[0], 400.0)
    viewer.trigger("window.robust")
    assert viewer.scene.base_layer().display.window == pytest.approx(src.robust_range(0))


def test_the_opening_window_is_written_once_and_holds_through_frames(qtbot, ds, no_gpu) -> None:
    """Recomputed per frame, the contrast flickered during playback."""
    viewer = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    qtbot.waitUntil(lambda: not viewer.jobs.busy(), timeout=10_000)
    viewer.qstore.flush()
    w0 = viewer.scene.base_layer().display.window
    assert w0 is not None
    viewer.trigger("frame.last")
    viewer.qstore.flush()
    assert viewer.scene.base_layer().display.window == w0


def test_pet_opens_with_a_window_for_the_whole_series(qtbot, tmp_path, no_gpu) -> None:
    """PET's first frame holds almost no counts: a window from it alone
    saturates every later frame."""
    data = np.zeros((10, 10, 10, 6), dtype=np.float32)
    data[..., 0] = np.random.default_rng(6).random((10, 10, 10))
    for t in range(1, 6):
        data[..., t] = np.random.default_rng(t).random((10, 10, 10)) * 100
    path = _write(tmp_path / "sub-01" / "pet" / "sub-01_pet.nii.gz", data)
    viewer = _open(qtbot, _viewer(qtbot), path, tmp_path)
    qtbot.waitUntil(lambda: viewer.scene.base_layer().display.window[1] > 10, timeout=10_000)


def test_a_window_the_user_chose_is_never_replaced(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    qtbot.waitUntil(lambda: not viewer.jobs.busy(), timeout=10_000)
    viewer.run("layer.set", window=(0.2, 0.3))
    viewer.presenter._apply_series_window((0.0, 9.0))
    assert viewer.scene.base_layer().display.window == (0.2, 0.3)


def test_a_colour_map_changes_what_is_drawn(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.axial")
    _settle(qtbot, viewer)
    canvas = _canvas(viewer, "axial")
    pt = canvas.grid_to_screen(2, 2).toPoint()    # away from the crosshair
    gray = canvas.grab_image().pixelColor(pt)
    combo = _column(viewer).control("colormap")
    combo.setCurrentIndex(combo.findData("hot"))
    _settle(qtbot, viewer)
    hot = canvas.grab_image().pixelColor(pt)
    assert viewer.scene.base_layer().display.colormap == "hot"
    assert gray.red() == gray.green() == gray.blue()
    assert hot.red() > hot.blue()


def test_display_changes_can_be_undone(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("layer.set", colormap="viridis")
    viewer.qstore.flush()
    assert viewer.action("edit.undo").isEnabled()
    viewer.trigger("edit.undo")
    viewer.qstore.flush()
    assert viewer.scene.base_layer().display.colormap == "gray"
    viewer.trigger("edit.redo")
    assert viewer.scene.base_layer().display.colormap == "viridis"


# ---------------------------------------------------------------------------
# Other layouts, the screenshot, keys, theme
# ---------------------------------------------------------------------------


def test_hero_shows_one_large_plane_and_two_small(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("view.mode", mode="hero")
    _settle(qtbot, viewer)
    canvases = viewer.canvases("slice")
    assert sorted(c.plane for c in canvases) == ["axial", "coronal", "sagittal"]
    hero = max(canvases, key=lambda c: c.width() * c.height())
    assert hero.plane == viewer.scene.plane
    assert hero.width() > 1.5 * min(c.width() for c in canvases)


def test_the_mosaic_draws_its_line(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("view.mode", mode="mosaic")
    viewer.run("view.mosaic_text", text="A 1 3 5 ; S 4")
    viewer.qstore.flush()
    (canvas,) = viewer.canvases("mosaic")
    image = canvas.grab().toImage()
    lit = sum(image.pixelColor(x, y).lightness() > 30
              for x in range(0, image.width(), 6) for y in range(0, image.height(), 6))
    assert lit > 20


def test_a_screenshot_is_a_png_of_the_view(qtbot, ds, no_gpu, tmp_path) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    out = viewer.save_screenshot(tmp_path / "shot.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_key_switches_the_plane_once_the_viewer_has_focus(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.activateWindow()
    viewer.setFocus()
    qtbot.waitUntil(viewer.hasFocus, timeout=2000)
    qtbot.keyClick(viewer, Qt.Key.Key_S)
    viewer.qstore.flush()
    assert (viewer.scene.mode, viewer.scene.plane) == ("single", "sagittal")


def test_a_rebound_key_applies_to_an_open_viewer(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    SettingsHub.instance().update(lambda s: s.keymap.__setitem__("view.sagittal", ["Shift+Q"]))
    assert viewer.action_manager.keys_for("view.sagittal") == ["Shift+Q"]
    assert keynames.key("Shift+Q") in viewer.action("view.sagittal").toolTip()


def test_the_help_lists_the_live_keys(qtbot, ds, no_gpu) -> None:
    from bidsmgr.gui.viz.help import help_html

    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    SettingsHub.instance().update(lambda s: s.keymap.__setitem__("view.graph", ["Shift+G"]))
    html = help_html(viewer.action_manager, {})
    assert keynames.key("Shift+G") in html and "Scroll" in html


@pytest.mark.parametrize("mac", [True, False])
def test_keys_are_named_as_the_os_names_them(qtbot, ds, no_gpu, monkeypatch, mac) -> None:
    """Qt binds Ctrl to Command on a Mac: there the help, the tooltips, the
    settings table and the side tab say Command, and nowhere "Ctrl"."""
    from bidsmgr.gui.viz.help import help_html
    from bidsmgr.gui.viz.settings_pages import ShortcutsPage

    monkeypatch.setattr(keynames, "is_mac", lambda: mac)
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.action_manager.apply_keymap({})
    html = help_html(viewer.action_manager, {})
    tip = viewer.action("view.inspector").toolTip()
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(SettingsHub.instance().settings)
    table = " ".join(page.table.item(r, 2).text() for r in range(page.table.rowCount()))
    gestures = " ".join(page.mouse_table.item(r, 1).text()
                        for r in range(page.mouse_table.rowCount()))
    side = viewer.presenter.side_tab
    side._sync_tip()
    if mac:
        for text in (html, tip, table, gestures, side.toolTip()):
            assert "Ctrl" not in text
        assert "⌘I" in tip and "⌘I" in side.toolTip()
        assert "⇧⌘Z" in table and "⌘ + Click / drag" in gestures
        assert "⌘" in html
    else:
        assert "Ctrl+I" in tip and "Ctrl+I" in side.toolTip()
        assert "Ctrl+Shift+Z" in table and "Ctrl + Click / drag" in gestures
        assert "⌘" not in html + tip + table + gestures


def test_a_search_for_a_key_name_finds_its_symbol(qtbot, ds, no_gpu, monkeypatch) -> None:
    from bidsmgr.gui.viz.help import ShortcutsDialog

    monkeypatch.setattr(keynames, "is_mac", lambda: True)
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    dlg = ShortcutsDialog(viewer, viewer.action_manager, {}, "volume")
    qtbot.addWidget(dlg)
    dlg.search.setText("cmd")
    rows = [w for card in dlg.cards if card.matches for w, _k in card.shown_rows]
    assert "Undo a display change" in rows


def test_the_theme_reaches_the_plots_but_not_the_image(qtbot, ds, no_gpu) -> None:
    from bidsmgr.gui import theme_manager
    from bidsmgr.gui.viz.bridge import ThemeHub

    viewer, graph = _with_graph(qtbot, ds)
    try:
        viewer.repaint_for_palette(theme_manager.LIGHT)
        assert graph.plot.backgroundBrush().color().name().lower() == \
            theme_manager.LIGHT["bg"].lower()
        viewer.trigger("view.axial")
        _settle(qtbot, viewer)
        corner = _canvas(viewer, "axial").grab_image().pixelColor(1, 1)
        assert (corner.red(), corner.green(), corner.blue()) == (0, 0, 0)
    finally:
        ThemeHub.instance().publish(theme_manager.DARK)


def test_a_closed_viewer_neither_breaks_a_theme_swap_nor_stays_in_memory(qtbot, ds, no_gpu) -> None:
    """The hubs live for the process; a viewer in a closed dialog does not.

    A lambda on a hub kept the dead viewer alive and called into its deleted
    canvases on the next theme swap; a callable on ``destroyed`` segfaulted
    inside the C++ destructor. Both are guarded here.
    """
    import gc
    import weakref

    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui import theme_manager
    from bidsmgr.gui.viz.bridge import ThemeHub

    viewer = Viewer(kind="volume")
    viewer.resize(800, 500)
    viewer.show()
    with qtbot.waitSignal(viewer.loaded, timeout=20_000):
        viewer.set_file(_bold(ds), ds)
    viewer.trigger("view.graph")
    viewer.qstore.flush()
    ref = weakref.ref(viewer)
    viewer.close()
    viewer.deleteLater()
    del viewer
    # PyQt deletes its slot proxies with deleteLater once their sender is
    # gone, so the deferred deletes take more than one round.
    for _ in range(4):
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        QApplication.processEvents()
    try:
        ThemeHub.instance().publish(theme_manager.LIGHT)
        SettingsHub.instance().update(lambda s: setattr(s.crosshair, "thickness", 2))
    finally:
        ThemeHub.instance().publish(theme_manager.DARK)
    gc.collect()
    assert ref() is None


def test_without_a_gpu_there_is_no_3d_to_offer(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.presenter.render_canvas is None
    assert not viewer.action("view.3d").isEnabled()
    assert not viewer.action("view.combo").isEnabled()
    # A state from a machine WITH a GPU is applied harmlessly: the nearest
    # layout this one can show is the three planes, not one slice.
    viewer.apply_state({"mode": "combo", "render": {"effect": "Glass"},
                        "clips": [{"active": True, "flip": True}]})
    viewer.qstore.flush()
    assert viewer.scene.mode == "multi"
    assert viewer.presenter.render_canvas is None


def test_a_viewer_destroyed_mid_read_neither_aborts_nor_leaves_work(qtbot, tmp_path, no_gpu) -> None:
    """Nobody calls stop_loading here, as nobody does when a parent widget
    is closed: the viewer never sees a close event. A running QThread
    destroyed with its owner used to abort the process."""
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.workers.viz import live_jobs

    big = _write(tmp_path / "sub-01_task-a_bold.nii.gz",
                 np.random.default_rng(1).random((48, 48, 32, 160)).astype(np.float32))
    viewer = Viewer(kind="volume")
    viewer.set_file(big, tmp_path)
    qtbot.waitUntil(lambda v=viewer: v.jobs.running("stream") or v.is_loaded(),
                    timeout=20_000)
    viewer.deleteLater()
    del viewer
    for _ in range(4):
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        QApplication.processEvents()
    # The orphaned read notices its owner is gone and stops at the next chunk.
    qtbot.waitUntil(lambda: live_jobs() == 0, timeout=20_000)


# ---------------------------------------------------------------------------
# The controls column (generated from viz.props and render3d.PARAMS)
# ---------------------------------------------------------------------------


class _Column:
    """The controls column, by what the tests reach for."""

    def __init__(self, viewer) -> None:
        if not viewer.presenter.inspector_open():
            viewer.trigger("view.inspector")
            viewer.qstore.flush()
        self.insp = viewer.presenter.inspector
        assert self.insp is not None and self.insp.isVisible()
        self.list = self.insp.section("layers").list
        self.view = self.insp.section("view")

    def control(self, key):
        for name in ("display", "overlay"):
            found = self.insp.section(name).control(key)
            if found is not None:
                return found
        raise KeyError(key)

    def minimumSizeHint(self):  # noqa: N802
        return self.insp.minimumSizeHint()


def _column(viewer) -> _Column:
    return _Column(viewer)


def _with_layers(qtbot, ds, path=None):
    viewer = _open(qtbot, _viewer(qtbot), path or _t1(ds), ds)
    return viewer, _column(viewer)


def test_the_column_edits_through_commands(qtbot, ds, no_gpu) -> None:
    viewer, panel = _with_layers(qtbot, ds)
    assert viewer.action("view.inspector").isChecked()
    combo = panel.control("colormap")
    combo.setCurrentIndex(combo.findData("hot"))
    panel.control("opacity").type_value(0.5)
    panel.control("invert").setChecked(True)
    d = viewer.scene.base_layer().display
    assert (d.colormap, d.opacity, d.invert) == ("hot", 0.5, True)
    # Each change is a command: undoable.
    viewer.qstore.flush()
    viewer.trigger("edit.undo")
    viewer.qstore.flush()
    assert viewer.scene.base_layer().display.invert is False


def test_the_column_follows_the_scene(qtbot, ds, no_gpu) -> None:
    viewer, panel = _with_layers(qtbot, ds)
    viewer.run("layer.set", colormap="viridis", gamma=2.0)
    viewer.qstore.flush()
    assert panel.control("colormap").currentData() == "viridis"
    assert panel.control("gamma").value() == 2.0


def test_the_negative_window_appears_with_a_negative_map(qtbot, ds, no_gpu) -> None:
    viewer, panel = _with_layers(qtbot, ds)
    assert not panel.control("window_negative").isVisible()
    panel.control("colormap_negative").setCurrentIndex(
        panel.control("colormap_negative").findData("winter"))
    viewer.qstore.flush()
    assert panel.control("window_negative").isVisible()


def test_hiding_a_layer_from_the_list(qtbot, ds, no_gpu) -> None:
    from PyQt6.QtCore import Qt as _Qt

    viewer, panel = _with_layers(qtbot, ds)
    panel.list.item(0).setCheckState(_Qt.CheckState.Unchecked)
    assert viewer.scene.base_layer().visible is False


def test_the_column_opens_by_default_and_remembers_being_closed(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.presenter.inspector_open()
    viewer.trigger("view.inspector")
    assert not SettingsHub.instance().settings.volume.inspector
    SettingsHub.reset_instance()
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert not viewer.presenter.inspector_open()


def test_sections_fold_and_remember_it(qtbot, ds, no_gpu) -> None:
    viewer, panel = _with_layers(qtbot, ds)
    view = panel.insp.section("view")
    assert view.is_open()
    view.header.clicked.emit()
    assert not view.is_open() and not view.body.isVisible()
    assert SettingsHub.instance().settings.inspector_sections["view"] is False
    SettingsHub.reset_instance()
    again = _column(_open(qtbot, _viewer(qtbot), _t1(ds), ds))
    assert not again.insp.section("view").is_open()


def test_a_screenshot_leaves_the_panel_out(qtbot, ds, no_gpu) -> None:
    viewer, panel = _with_layers(qtbot, ds)
    side = viewer.presenter.side
    figure = viewer.grab_figure()
    assert figure.width() + side.width() <= viewer.presenter.content.width() + 4


def test_a_viewer_made_only_to_render_has_no_column(qtbot, ds, no_gpu) -> None:
    viewer = Viewer(kind="volume", panels=False)
    qtbot.addWidget(viewer)
    viewer.resize(800, 500)
    viewer.show()
    qtbot.waitExposed(viewer)
    with qtbot.waitSignal(viewer.loaded, timeout=20_000):
        viewer.set_file(_t1(ds), ds)
    assert not viewer.presenter.inspector_open()


def test_an_unchosen_negative_window_shows_what_is_drawn(qtbot, ds, no_gpu) -> None:
    viewer, panel = _with_layers(qtbot, ds)
    viewer.run("layer.set", window=(10.0, 500.0), colormap_negative="winter")
    viewer.qstore.flush()
    rc = panel.control("window_negative")
    assert (rc.lo_spin.value(), rc.hi_spin.value()) == (10.0, 500.0)
    assert rc.window == (10.0, 500.0)


def test_the_graph_exports_its_series_as_csv(qtbot, ds, no_gpu, tmp_path) -> None:
    import csv

    viewer, graph = _with_graph(qtbot, ds)
    viewer.run("graph.set", scope=2, scaling="percent")
    viewer.qstore.flush()
    out = tmp_path / "tc.csv"
    graph.export_csv(out)
    rows = list(csv.reader(out.open()))
    i, j, k = views.cursor_voxel(viewer.store)
    assert rows[0][0] == "volume" and rows[0][1] == f"voxel {i} {j} {k} (percent change)"
    assert len(rows[0]) == 1 + graph.cell_count() and len(rows) == 1 + 3
    assert float(rows[3][1]) == pytest.approx(graph.center_series()[2], rel=1e-4)


def test_blocky_pixels_toggle(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.nearest")
    viewer.qstore.flush()
    assert viewer.scene.base_layer().display.interpolation == "nearest"
    assert viewer.action("view.nearest").isChecked()
    viewer.trigger("view.nearest")
    assert viewer.scene.base_layer().display.interpolation == "linear"


def test_a_transparent_figure_has_no_surround(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.trigger("view.axial")
    _settle(qtbot, viewer)
    opaque = viewer.grab_figure(scale=1.0)
    clear = viewer.grab_figure(scale=1.0, transparent=True)
    assert opaque.pixelColor(2, 2).alpha() == 255
    assert clear.pixelColor(2, 2).alpha() == 0
    # The image itself is still drawn.
    canvas = _canvas(viewer, "axial")
    centre = canvas.mapTo(viewer.presenter.figure_widget(), canvas.grid_to_screen(3, 3).toPoint())
    assert clear.pixelColor(centre).alpha() == 255


class TestShrinking:
    """Opening the graph made the viewer 1463 x 428 px at least: its controls
    were one row whose width was the sum of its parts, and the physio strip
    asked 22 px a lane."""

    def test_the_graph_does_not_pin_the_viewer_wide(self, qtbot, ds, no_gpu):
        viewer, graph = _with_graph(qtbot, ds)
        viewer.presenter.set_inspector(False, remember=False)
        # The layout settles on the next turn of the event loop (a hidden
        # column's room is given back then, not at once).
        qtbot.wait(20)
        hint = viewer.minimumSizeHint()
        assert hint.width() < 420 and hint.height() < 360

    def test_wrapped_controls_are_never_painted_over(self, qtbot, ds, no_gpu):
        viewer, graph = _with_graph(qtbot, ds)
        viewer.presenter.set_inspector(False, remember=False)
        viewer.resize(380, 600)
        qtbot.wait(50)
        bar = graph.controls
        assert bar.height() >= bar.heightForWidth(bar.width())
        assert bar.minimumSizeHint().height() == bar.heightForWidth(bar.width())

    def test_the_controls_column_gives_way_and_comes_back(self, qtbot, ds, no_gpu):
        viewer, _graph = _with_graph(qtbot, ds)
        viewer.resize(1200, 700)
        viewer.presenter.set_inspector(True, remember=False)
        qtbot.wait(20)
        assert viewer.presenter.inspector_open()
        viewer.resize(520, 700)
        qtbot.waitUntil(lambda: not viewer.presenter.inspector_open(), timeout=2000)
        viewer.resize(1300, 700)
        SettingsHub.instance().update(lambda s: setattr(s.volume, "inspector", True),
                                      persist=False)
        viewer.resize(1320, 700)
        qtbot.waitUntil(viewer.presenter.inspector_open, timeout=2000)


class TestEverythingPersists:
    """Point 1 of round 4: a new window or file kept the mode and plane but
    lost the layout, the 3-D look, the clip planes and the crosshair flag."""

    def test_a_new_window_opens_with_the_layout_left(self, qtbot, ds, no_gpu):
        first = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        first.run("view.mode", mode="hero")
        first.run("layout.set", arrangement="grid", hero_fraction=0.7, hero_side="top",
                  planes=["axial", "coronal"])
        first.trigger("view.crosshair")
        first.qstore.flush()
        SettingsHub.reset_instance()
        second = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        lay = second.scene.layout
        assert second.scene.mode == "hero"
        assert (lay.arrangement, lay.hero_fraction, lay.hero_side) == ("grid", 0.7, "top")
        assert lay.planes == ["axial", "coronal"]
        assert second.scene.display.crosshair is False

    def test_the_3d_look_and_clip_planes_persist(self, qtbot, ds, no_gpu):
        first = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        first.run("render.effect", effect="MIP")
        first.run("clip.preset", preset="corner")
        first.qstore.flush()
        first.presenter.stop()            # what settles is written on close
        SettingsHub.reset_instance()
        second = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        assert second.scene.render.effect == "MIP"
        assert [c.active for c in second.scene.clips] == [c.active for c in first.scene.clips]

    def test_the_next_file_keeps_the_layout_and_mosaic(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        viewer.run("layout.set", arrangement="column")
        viewer.run("view.mosaic_text", text="A -10 0 10")
        viewer.qstore.flush()
        _open(qtbot, viewer, _t2(ds), ds)
        assert viewer.scene.layout.arrangement == "column"
        assert viewer.scene.mosaic == "A -10 0 10"

    def test_the_comparison_mirrors_the_layout(self, qtbot, ds, no_gpu):
        from bidsmgr.gui.viz.compare import ComparePanes

        panes = ComparePanes()
        qtbot.addWidget(panes)
        panes.resize(1100, 600)
        panes.show()
        with qtbot.waitSignal(panes.both_loaded, timeout=20_000):
            panes.show_images(_t1(ds), _t2(ds), root=ds)
        panes.left.run("layout.set", arrangement="row", planes=["sagittal"])
        panes.left.qstore.flush()
        assert panes.right.scene.layout.arrangement == "row"
        assert panes.right.scene.layout.planes == ["sagittal"]


class TestMosaicBuilder:
    def test_the_builder_writes_a_grid_fitted_to_the_head(self, qtbot, ds, no_gpu):
        from bidsmgr.viz.compute import mosaic as M

        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        viewer.run("mosaic.set", plane="axial", rows=2, cols=3, fit=True)
        viewer.run("view.mode", mode="mosaic")
        viewer.qstore.flush()
        spec = M.parse(viewer.scene.mosaic)
        assert [len(r) for r in spec.rows] == [3, 3]
        assert {t.plane for t in spec.tiles} == {"axial"}
        mm = [t.mm for t in spec.tiles]
        assert mm == sorted(mm) and len(set(mm)) == 6
        (canvas,) = viewer.canvases("mosaic")
        canvas.repaint()
        assert not canvas.grab().isNull()

    def test_a_line_written_by_hand_is_left_alone(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        viewer.run("view.mosaic_text", text="A 1 3 ; S 4")
        viewer.run("view.mode", mode="mosaic")
        viewer.qstore.flush()
        assert viewer.scene.mosaic == "A 1 3 ; S 4", "not refitted on entering the mosaic"
        viewer.run("mosaic.set", cols=2)
        assert viewer.scene.mosaic != "A 1 3 ; S 4", "a control takes over again"

    def test_the_section_drives_it(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        viewer.presenter.set_inspector(True)
        section = viewer.presenter.inspector.section("mosaic")
        section.rows.type_value(1)
        section.cols.type_value(4)
        viewer.qstore.flush()
        assert (viewer.scene.mosaic_build.rows, viewer.scene.mosaic_build.cols) == (1, 4)
        section.reference.setChecked(True)
        viewer.qstore.flush()
        assert viewer.scene.mosaic.startswith("S X")

    def test_the_figure_is_saved_at_the_chosen_scale(self, qtbot, ds, no_gpu, tmp_path):
        from PyQt6.QtGui import QImage

        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        out = tmp_path / "fig.png"
        assert viewer.presenter.save_mosaic_figure(out, scale=2.0, transparent=True) == out
        assert viewer.scene.mode == "mosaic"
        img = QImage(str(out))
        assert img.width() >= 2 * viewer.canvases("mosaic")[0].width() - 2
        assert img.hasAlphaChannel()


class TestRestoringDefaults:
    def test_a_section_restores_its_own_defaults(self, qtbot, ds, no_gpu):
        from bidsmgr.viz.scene import LayoutState

        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        viewer.presenter.set_inspector(True)
        viewer.run("layout.set", arrangement="grid", hero_fraction=0.8)
        viewer.run("view.flag", flag="radiological")
        viewer.qstore.flush()
        insp = viewer.presenter.inspector
        assert insp.section("layout").header.resettable
        assert not insp.section("layers").header.resettable, "the layers are content"
        insp.section("layout").header.reset_clicked.emit()
        assert viewer.scene.layout == LayoutState()
        assert viewer.scene.display.radiological, "the other sections untouched"
        viewer.store.undo()
        assert viewer.scene.layout.arrangement == "grid"

    def test_the_layer_display_goes_back_to_how_it_opened(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        viewer.presenter.set_inspector(True)
        viewer.run("layer.set", colormap="hot", gamma=3.0)
        viewer.qstore.flush()
        viewer.presenter.inspector.section("display").restore_defaults()
        d = viewer.scene.base_layer().display
        assert (d.colormap, d.gamma) == ("gray", 1.25)

    def test_everything_at_once_keeps_the_users_own_work(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        viewer.presenter.save_view("Mine")
        viewer.run("view.mode", mode="hero")
        viewer.run("render.effect", effect="MIP")
        SettingsHub.instance().update(lambda s: s.keymap.update({"view.axial": ["Z"]}))
        viewer.qstore.flush()
        assert viewer.presenter.restore_all_defaults(confirm=False)
        s = SettingsHub.instance().settings
        assert "Mine" in s.view_presets and s.keymap == {"view.axial": ["Z"]}
        assert s.layout_state == {} or viewer.scene.mode != "hero"
        assert viewer.scene.render.effect != "MIP"


class TestToolbarByPurpose:
    def test_one_layout_menu_holds_planes_and_layouts(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        texts = [a.text() for a in viewer.presenter.layout_button.menu().actions() if a.text()]
        assert texts[:3] == ["Axial view", "Coronal view", "Sagittal view"]
        assert "Mosaic of many slices (lightbox)" in texts
        assert viewer.button("view.axial") is None, "no separate plane buttons"

    def test_the_side_tab_opens_and_closes_the_controls(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        viewer.presenter.set_inspector(False)
        tab = viewer.presenter.side_tab
        assert tab.isVisibleTo(viewer) and not tab.open
        tab.clicked.emit()
        assert viewer.presenter.inspector_open() and tab.open
        tab.clicked.emit()
        assert not viewer.presenter.inspector_open()

    def test_the_save_menu_names_what_it_keeps(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        menu = viewer.presenter.save_button.menu()
        viewer.presenter._fill_save_menu(menu)
        texts = [a.text() for a in menu.actions() if a.text()]
        assert texts == ["Save a screenshot...", "Save a mosaic figure...",
                         "Save the look as a preset...", "Save a scene in this dataset...",
                         "Show the command line...", "Restore every viewer default..."]
        for a in menu.actions():
            if a.text():
                assert a.toolTip() and a.toolTip() != a.text().rstrip(".")

    def test_a_3d_image_has_no_volume_stepper(self, qtbot, ds, no_gpu):
        """Time course, Play and Volume mean nothing for one volume: they are
        not there, rather than greyed."""
        viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        assert viewer.presenter._frame_group.isHidden()
        assert not viewer.button("view.graph").isVisibleTo(viewer)
        assert not viewer.button("frame.play").isVisibleTo(viewer)
        _open(qtbot, viewer, _bold(ds), ds)
        assert not viewer.presenter._frame_group.isHidden()
        assert viewer.button("view.graph").isVisibleTo(viewer)


class TestMosaicCrop:
    """Tiles cropped to the head, ONE crop per plane so they stay the same
    size and a structure sits at the same place in each."""

    def _head(self, ds) -> Path:
        # A small bright block in a large empty field of view.
        data = np.zeros((40, 40, 20), dtype=np.float32)
        data[15:25, 12:28, 4:16] = 100.0
        return _write(ds / "sub-01" / "anat" / "sub-01_PD.nii.gz", data)

    def test_the_content_box_and_the_shared_crop(self):
        from bidsmgr.gui.viz.canvases.mosaic import content_box, shared_crop

        img = np.zeros((10, 20, 4), dtype=np.uint8)
        assert content_box(img) is None
        img[2:5, 4:10, :3] = 200
        assert content_box(img) == (0.2, 0.2, 0.5, 0.5)
        crop = shared_crop([(0.2, 0.2, 0.5, 0.5), (0.3, 0.1, 0.6, 0.4), None])
        assert crop[0] < 0.2 and crop[1] < 0.1 and crop[2] > 0.6 and crop[3] > 0.5
        assert shared_crop([None]) == (0.0, 0.0, 1.0, 1.0)

    def test_tiles_are_cropped_unless_asked_not_to(self, qtbot, ds, no_gpu):
        viewer = _open(qtbot, _viewer(qtbot), self._head(ds), ds)
        viewer.run("mosaic.set", plane="axial", rows=1, cols=3, fit=False)
        viewer.run("view.mode", mode="mosaic")
        viewer.qstore.flush()
        (canvas,) = viewer.canvases("mosaic")
        left, top, right, bottom = canvas.crops(canvas.tile_rows())["axial"]
        assert (right - left) < 0.8 and (bottom - top) < 0.8, "the empty field was kept"
        viewer.run("mosaic.set", crop=False)
        viewer.qstore.flush()
        assert canvas.crops(canvas.tile_rows())["axial"] == (0.0, 0.0, 1.0, 1.0)
        canvas.repaint()
        assert not canvas.grab().isNull()


def test_a_series_is_shown_only_once_it_is_read(qtbot, ds, no_gpu) -> None:
    """Volumes appearing one by one, the graph growing and the contrast
    changing as they arrived read as something going wrong."""
    viewer = _viewer(qtbot)
    pages = []
    viewer.ctx.jobs.progressed.connect(
        lambda tag, _gen, done, total: pages.append(viewer.page()) if tag == "stream" else None)
    _open(qtbot, viewer, _bold(ds), ds)
    assert viewer.is_loaded() and viewer.page() == "content"
    assert pages and set(pages) == {"loading"}, "the series showed before it was read"


def test_a_saved_view_with_rows_that_no_longer_exist_still_works(qtbot, ds, no_gpu) -> None:
    """A view saved by round 4 named the row ``motion``; one unknown id made
    every change to the rows fail, so FD and motion could not be shown."""
    SettingsHub.instance().update(lambda s: s.layout_state.update(
        {"mri.func": {"mode": "multi", "graph_visible": True,
                      "graph": {"qc": True, "qc_rows": ["motion", "dvars"]}}}))
    viewer = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    assert viewer.scene.graph.qc_rows == ["dvars"]
    viewer.run("graph.set", qc_rows=["fd", "dvars", "rotation"])
    assert viewer.scene.graph.qc_rows == ["fd", "dvars", "rotation"]
