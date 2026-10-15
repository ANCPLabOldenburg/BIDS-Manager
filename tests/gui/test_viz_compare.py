"""Two images side by side, driven as one: the compare and defacing dialogs.

Replaces ``test_compare_dialog``, ``test_compare_no_gpu`` and the 2-D half of
``test_deface_compare``. No GPU anywhere in this file (its 3-D twin is
``test_viz_compare_3d``): creating a viewer that pre-builds an OpenGL widget
and then one that does not, in one process, destabilised Qt's offscreen
platform, and a machine either has a GPU or it does not.

The crosshair travels as MILLIMETRES, so the tests compare world positions,
and they compare voxel indices only where the grids are the same.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.deface import status  # noqa: E402
from bidsmgr.deface.engines import ALLINEATE, TEMPLATE  # noqa: E402
from bidsmgr.gui.compare_dialog import CompareDialog, is_nifti, open_compare  # noqa: E402
from bidsmgr.gui.deface_compare import DefaceCompareDialog  # noqa: E402
from bidsmgr.gui.viz.bridge import SettingsHub  # noqa: E402
from bidsmgr.project.operations import begin_operation  # noqa: E402
from bidsmgr.viz import views  # noqa: E402

pytestmark = pytest.mark.gui

REL = "sub-01/anat/sub-01_T1w.nii.gz"


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    """Patched at the source: the viewer probes the GPU when it is built."""
    from bidsmgr.gui.viz.canvases import render

    monkeypatch.setattr(render, "gpu_available", lambda: False)


# ---------------------------------------------------------------------------
# Data and dialogs
# ---------------------------------------------------------------------------


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"}))
    shutil.copyfile(TEMPLATE, root / REL)
    return root


def _defaced(tmp_path: Path) -> Path:
    """A dataset whose image was defaced, so an original is kept."""
    root = _dataset(tmp_path)
    with begin_operation(root, "Deface 1 image (allineate)") as op:
        op.write_bytes(root / REL, (root / REL).read_bytes())
    status.sidecar_for(root / REL).write_text(json.dumps(status.record({}, ALLINEATE)))
    return root


@pytest.fixture
def images(tmp_path: Path):
    root = _dataset(tmp_path)
    a = root / REL
    b = root / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
    shutil.copyfile(TEMPLATE, b)
    return root, a, b


@pytest.fixture
def dialog(qtbot):
    """Build a dialog and CLOSE it afterwards however the test ends.

    Closing is not tidiness: both viewers read on their own QThread, and a
    dialog destroyed while a read is in flight takes the process with it.
    """
    made = []

    def _make(cls, *args, **kwargs):
        dlg = cls(*args, **kwargs)
        qtbot.addWidget(dlg)
        made.append(dlg)
        return dlg

    yield _make
    for dlg in made:
        dlg.close()


def _both_loaded(qtbot, panes) -> None:
    qtbot.waitUntil(lambda: panes.left.is_loaded() and panes.right.is_loaded()
                    and panes.link.isEnabled(), timeout=20_000)
    panes.left.qstore.flush()
    panes.right.qstore.flush()


def _move(viewer, **delta) -> tuple:
    """Move the crosshair by whole voxels, the way a click or a key does."""
    vox = list(views.cursor_voxel(viewer.store))
    for axis, d in delta.items():
        k = "ijk".index(axis)
        vox[k] = max(0, vox[k] + d)
    viewer.run("cursor.set_voxel", i=vox[0], j=vox[1], k=vox[2])
    viewer.qstore.flush()
    return tuple(vox)


def _world(viewer) -> np.ndarray:
    return np.asarray(viewer.scene.cursor.world, dtype=float)


# ---------------------------------------------------------------------------
# The general compare dialog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("a.nii", True), ("a.nii.gz", True), ("A.NII.GZ", True),
    ("a.json", False), ("a.tsv", False), ("a.niftii", False),
])
def test_what_counts_as_an_image(name, expected) -> None:
    assert is_nifti(Path(name)) is expected


def test_two_images_open_side_by_side(qtbot, dialog, images) -> None:
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    _both_loaded(qtbot, dlg.panes)
    assert dlg.panes.left.current_file() == a
    assert dlg.panes.right.current_file() == b


def test_it_opens_with_nothing_chosen(dialog, images) -> None:
    """A Tools entry has no selection to work from, so it must still open."""
    root, _a, _b = images
    dlg = dialog(CompareDialog, root=root)
    assert dlg.panes.left.current_file() is None
    assert all("nothing chosen" in c for c in dlg.captions())


def test_one_image_waits_on_the_left(dialog, images) -> None:
    root, a, _b = images
    dlg = dialog(CompareDialog, a, root=root)
    assert dlg.panes.left.current_file() is None, "half a pair was shown"
    assert a.name in dlg.captions()[0]


def test_captions_are_dataset_relative(dialog, images) -> None:
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    assert dlg.captions()[0] == "sub-01/anat/sub-01_T1w.nii.gz"


def test_captions_fall_back_to_the_name_outside_a_dataset(dialog, images, tmp_path) -> None:
    _root, a, _b = images
    stray = tmp_path / "elsewhere.nii.gz"
    shutil.copyfile(TEMPLATE, stray)
    dlg = dialog(CompareDialog, a, stray, root=None)
    assert dlg.captions()[1] == "elsewhere.nii.gz"


def test_open_compare_takes_the_selection_and_skips_non_images(qtbot, images) -> None:
    root, a, b = images
    dlg = open_compare(None, [a, b], root=root)
    qtbot.addWidget(dlg)
    assert dlg.chosen == (a, b)
    dlg.close()
    dlg = open_compare(None, [root / "dataset_description.json", a], root=root)
    qtbot.addWidget(dlg)
    assert dlg.chosen == (a, None)
    dlg.close()


def test_the_crosshair_is_linked(qtbot, dialog, images) -> None:
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    _both_loaded(qtbot, dlg.panes)
    moved = _move(dlg.panes.left, i=-5)
    assert views.cursor_voxel(dlg.panes.right.store) == moved
    # And from the right, too: the link runs both ways.
    moved = _move(dlg.panes.right, k=3)
    assert views.cursor_voxel(dlg.panes.left.store) == moved


def test_different_shapes_are_linked_through_the_scanner(qtbot, dialog, images) -> None:
    """A cropped image shares no voxel index with its source and is still a
    picture of the same head."""
    root, a, _b = images
    cropped = root / "sub-01" / "anat" / "sub-01_desc-crop_T1w.nii.gz"
    nib.save(nib.load(str(TEMPLATE)).slicer[:, :, 8:], str(cropped))
    dlg = dialog(CompareDialog, a, cropped, root=root)
    panes = dlg.panes
    _both_loaded(qtbot, panes)
    assert panes.link.isChecked()
    assert "Different sizes" in panes.note.text() and "scanner" in panes.note.text()
    _move(panes.left, k=-6)
    assert np.allclose(_world(panes.left), _world(panes.right), atol=1.01)
    assert views.cursor_voxel(panes.right.store) != views.cursor_voxel(panes.left.store), \
        "the indices happen to match, so this no longer tests world mapping"
    panes.left.trigger("view.sagittal")
    panes.left.qstore.flush()
    assert (panes.right.scene.mode, panes.right.scene.plane) == ("single", "sagittal")


def test_one_toolbar_while_synced(qtbot, dialog, images) -> None:
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    dlg.show()
    _both_loaded(qtbot, dlg.panes)
    assert dlg.panes.left.toolbar_visible()
    assert not dlg.panes.right.toolbar_visible()
    dlg.panes.link.setChecked(False)
    assert dlg.panes.right.toolbar_visible(), "unsyncing gave the right image no controls"
    dlg.panes.link.setChecked(True)
    assert not dlg.panes.right.toolbar_visible()


def test_closing_mid_load_stops_both_reads(dialog, images) -> None:
    """A running read destroyed with its viewer aborts the process."""
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    dlg.close()
    assert not dlg.panes.left.jobs.busy()
    assert not dlg.panes.right.jobs.busy()


def test_the_window_fits_the_screen(dialog, images) -> None:
    from PyQt6.QtWidgets import QApplication

    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    screen = dlg.screen() or QApplication.primaryScreen()
    if screen is None:
        pytest.skip("no screen")
    assert dlg.width() <= screen.availableGeometry().width()
    assert dlg.height() <= screen.availableGeometry().height()


def test_the_4d_graph_and_its_options_follow(qtbot, dialog, tmp_path) -> None:
    """The old pane announced the state it was LEAVING, so the other one
    showed "graph off" while the user turned it on."""
    root = tmp_path / "ds"
    data = np.random.default_rng(0).random((12, 12, 8, 6)).astype("float32")
    paths = []
    for task in ("x", "y"):
        p = root / "sub-01" / "func" / f"sub-01_task-{task}_bold.nii.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
        paths.append(p)
    dlg = dialog(CompareDialog, *paths, root=root)
    panes = dlg.panes
    _both_loaded(qtbot, panes)
    # A BOLD pair opens with the graph; turning it off and on follows.
    assert panes.left.scene.graph_visible and panes.right.scene.graph_visible
    panes.left.trigger("view.graph")
    panes.left.qstore.flush()
    assert not panes.right.scene.graph_visible
    panes.left.trigger("view.graph")
    panes.left.qstore.flush()
    assert panes.right.scene.graph_visible
    panes.left.run("graph.set", scope=3, mark_neighbors=False)
    panes.left.qstore.flush()
    assert panes.right.scene.graph.scope == 3
    assert panes.right.scene.graph.mark_neighbors is False
    panes.left.run("frame.set", frame=3)
    panes.left.qstore.flush()
    assert panes.right.scene.base_layer().frame == 3


def test_contrast_is_independent_unless_asked(qtbot, dialog, images) -> None:
    """Two different contrasts of one head want their own windows."""
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    panes = dlg.panes
    _both_loaded(qtbot, panes)
    assert not panes.contrast.isChecked()
    panes.left.run("layer.set", window=(0.0, 5.0), colormap="hot")
    panes.left.qstore.flush()
    assert panes.right.scene.base_layer().display.colormap == "gray"
    panes.contrast.setChecked(True)
    assert panes.right.scene.base_layer().display.colormap == "hot"
    assert panes.right.scene.base_layer().display.window == (0.0, 5.0)


# -- no GPU -----------------------------------------------------------------


def test_without_a_gpu_the_first_pair_opens_multi_planar(qtbot, dialog, images) -> None:
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    _both_loaded(qtbot, dlg.panes)
    assert dlg.panes.left.scene.mode == "multi"
    assert dlg.panes.left.presenter.render_canvas is None


def test_a_state_from_a_gpu_machine_is_applied_harmlessly(qtbot, dialog, images) -> None:
    """The same dataset gets opened on a workstation and on a laptop."""
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    _both_loaded(qtbot, dlg.panes)
    state = dlg.panes.left.state()
    state.update({"mode": "combo",
                  "render": {"effect": "Glass", "params": {},
                             "camera": {"az": 0.4, "el": 0.2, "dist": 2.0,
                                        "target": [0, 0, 0]}},
                  "clips": [{"active": True, "flip": True, "pos": 0.3}]})
    dlg.panes.right.apply_state(state)
    dlg.panes.right.qstore.flush()
    assert dlg.panes.right.scene.mode == "multi"
    assert dlg.panes.right.presenter.render_canvas is None


def test_a_remembered_3d_layout_falls_back_to_the_planes(qtbot, dialog, images) -> None:
    SettingsHub.instance().update(lambda s: setattr(s.volume, "mode", "combo"))
    SettingsHub.reset_instance()
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    _both_loaded(qtbot, dlg.panes)
    assert dlg.panes.left.scene.mode == "multi"


# ---------------------------------------------------------------------------
# The defacing dialog
# ---------------------------------------------------------------------------


def test_the_two_sides_are_the_two_versions(dialog, tmp_path) -> None:
    root = _defaced(tmp_path)
    dlg = dialog(DefaceCompareDialog, root, REL)
    assert dlg.panes.right.current_file() == root / REL
    assert dlg.panes.left.current_file() != root / REL
    assert dlg.panes.left.current_file().name == Path(REL).name


def test_before_and_after_share_one_contrast(qtbot, dialog, tmp_path) -> None:
    """The same image twice: a window that differs is a difference that is
    not in the data."""
    dlg = dialog(DefaceCompareDialog, _defaced(tmp_path), REL)
    panes = dlg.panes
    _both_loaded(qtbot, panes)
    assert panes.contrast.isChecked()
    panes.left.run("layer.set", window=(10.0, 400.0), gamma=2.0)
    panes.left.qstore.flush()
    d = panes.right.scene.base_layer().display
    assert d.window == (10.0, 400.0) and d.gamma == 2.0


def test_the_crosshair_follows_and_unlinking_stops_it(qtbot, dialog, tmp_path) -> None:
    dlg = dialog(DefaceCompareDialog, _defaced(tmp_path), REL)
    panes = dlg.panes
    _both_loaded(qtbot, panes)
    moved = _move(panes.left, i=-5)
    assert views.cursor_voxel(panes.right.store) == moved
    panes.link.setChecked(False)
    before = views.cursor_voxel(panes.right.store)
    _move(panes.left, i=-7)
    assert views.cursor_voxel(panes.right.store) == before


def test_a_cropped_result_still_links_and_says_how(qtbot, dialog, tmp_path) -> None:
    """The neck-cropping engine legitimately produces a smaller image."""
    root = _defaced(tmp_path)
    nib.save(nib.load(str(TEMPLATE)).slicer[:, :, 5:], str(root / REL))
    dlg = dialog(DefaceCompareDialog, root, REL)
    panes = dlg.panes
    _both_loaded(qtbot, panes)
    assert panes.link.isChecked()
    assert "Different sizes" in panes.note.text() and "scanner" in panes.note.text()
    _move(panes.left, k=-4)
    assert np.allclose(_world(panes.left), _world(panes.right), atol=1.01)
    # The view follows too: only voxel indices are meaningless across shapes.
    started = panes.right.scene.mode
    wanted = "single" if started != "single" else "multi"
    panes.left.run("view.mode", mode=wanted)
    panes.left.qstore.flush()
    assert panes.right.scene.mode == wanted


def test_closing_the_defacing_dialog_mid_load_stops_both_reads(dialog, tmp_path) -> None:
    dlg = dialog(DefaceCompareDialog, _defaced(tmp_path), REL)
    dlg.close()
    assert not dlg.panes.left.jobs.busy() and not dlg.panes.right.jobs.busy()


def test_no_original_explains_and_offers_no_restore(dialog, tmp_path) -> None:
    root = _dataset(tmp_path)
    status.sidecar_for(root / REL).write_text(json.dumps(status.record({}, ALLINEATE)))
    dlg = dialog(DefaceCompareDialog, root, REL)
    assert dlg.panes is None
    assert "No undefaced copy" in dlg._header.text()
    assert "during conversion" in dlg._subhead.text()
    assert not hasattr(dlg, "_restore")


def test_the_header_names_the_engine_and_the_source(dialog, tmp_path) -> None:
    dlg = dialog(DefaceCompareDialog, _defaced(tmp_path), REL)
    assert REL in dlg._header.text()
    text = dlg._subhead.text()
    assert ALLINEATE.label in text
    assert "edit history" in text or "you ran" in text


def test_restoring_is_offered_once_both_are_read(qtbot, dialog, tmp_path) -> None:
    dlg = dialog(DefaceCompareDialog, _defaced(tmp_path), REL)
    _both_loaded(qtbot, dlg.panes)
    qtbot.waitUntil(dlg._restore.isEnabled, timeout=5000)


def test_one_toolbar_and_resyncing_adopts_the_left(qtbot, dialog, tmp_path) -> None:
    dlg = dialog(DefaceCompareDialog, _defaced(tmp_path), REL)
    dlg.show()
    panes = dlg.panes
    _both_loaded(qtbot, panes)
    assert not panes.right.toolbar_visible()
    started = panes.right.scene.mode
    wanted = "single" if started != "single" else "multi"
    panes.link.setChecked(False)
    assert panes.right.toolbar_visible()
    panes.left.run("view.mode", mode=wanted)
    panes.left.qstore.flush()
    assert panes.right.scene.mode == started, "it followed with sync off"
    panes.link.setChecked(True)
    panes.right.qstore.flush()
    assert panes.right.scene.mode == wanted, "re-syncing left them apart"


def test_a_plane_button_and_a_key_switch_both(qtbot, dialog, tmp_path) -> None:
    """Through the BUTTON, not the setter underneath it."""
    dlg = dialog(DefaceCompareDialog, _defaced(tmp_path), REL)
    dlg.show()
    panes = dlg.panes
    _both_loaded(qtbot, panes)
    panes.left.action("view.sagittal").trigger()
    panes.left.qstore.flush()
    assert panes.right.scene.plane == "sagittal" and panes.right.scene.mode == "single"
    panes.left.trigger("view.coronal")
    panes.left.qstore.flush()
    assert panes.right.scene.plane == "coronal"


def test_the_panes_can_be_made_narrow(qtbot, dialog, tmp_path) -> None:
    """Two viewers used to pin the dialog at about 1750 px and grow it."""
    dlg = dialog(DefaceCompareDialog, _defaced(tmp_path), REL)
    _both_loaded(qtbot, dlg.panes)
    assert dlg.panes.left.minimumSizeHint().width() < 420
    assert dlg.panes.right.minimumSizeHint().width() < 420


# ---------------------------------------------------------------------------
# Symmetry: neither half is the main one
# ---------------------------------------------------------------------------


def _shown(qtbot, dialog, images):
    root, a, b = images
    dlg = dialog(CompareDialog, a, b, root=root)
    dlg.resize(1300, 760)
    dlg.show()
    _both_loaded(qtbot, dlg.panes)
    return dlg.panes


def test_one_toolbar_above_both_and_identical_headers(qtbot, dialog, images) -> None:
    panes = _shown(qtbot, dialog, images)
    bar = panes.left._toolbar
    assert bar.isVisible() and not panes.left.isAncestorOf(bar), "the toolbar sits in one half"
    assert bar.geometry().width() > panes.left.width(), "it does not span both halves"
    assert panes._left_head.change.isVisible() and panes._right_head.change.isVisible()
    assert abs(panes.left.width() - panes.right.width()) <= 3
    assert abs(panes.left.height() - panes.right.height()) <= 1


def test_one_controls_column_beside_both(qtbot, dialog, images) -> None:
    panes = _shown(qtbot, dialog, images)
    panes.left.presenter.set_inspector(True)
    assert panes.column_open()
    side = panes.left.presenter.side
    assert not panes.left.isAncestorOf(side), "the column squeezes one image"
    assert abs(panes.left.width() - panes.right.width()) <= 3
    # The right image's own key opens and closes the same column.
    panes.right.trigger("view.inspector")
    assert not panes.column_open()
    panes.right.trigger("view.inspector")
    assert panes.column_open()
    assert panes.right.presenter.side.isHidden(), "a second column appeared"


def test_the_column_shows_either_images_settings(qtbot, dialog, images) -> None:
    panes = _shown(qtbot, dialog, images)
    panes.left.presenter.set_inspector(True)
    panes.whose_buttons["right"].click()
    assert panes.column_open()
    assert panes.right.presenter.inspector is not None
    assert not panes.right.presenter.side.isHidden()
    assert not panes.right.isAncestorOf(panes.right.presenter.side)
    assert panes.left.presenter.side.isHidden()
    panes.whose_buttons["left"].click()
    assert panes.right.presenter.side.isHidden() and not panes.left.presenter.side.isHidden()


def test_unsyncing_gives_each_image_its_own_and_resyncing_takes_them_back(
        qtbot, dialog, images) -> None:
    panes = _shown(qtbot, dialog, images)
    panes.left.presenter.set_inspector(True)
    panes.link.setChecked(False)
    for viewer in (panes.left, panes.right):
        assert viewer.isAncestorOf(viewer._toolbar) and viewer.toolbar_visible()
        assert viewer.isAncestorOf(viewer.presenter.side)
        assert not viewer.presenter.side.isHidden(), "the open column was not carried"
        assert viewer.presenter.side_tab.isVisibleTo(viewer)
    assert not panes.column_open()
    panes.link.setChecked(True)
    assert panes.column_open()
    assert not panes.left.isAncestorOf(panes.left._toolbar)
    assert panes.right.presenter.side.isHidden()
    assert not panes.right.presenter.side_tab.isVisibleTo(panes.right)
