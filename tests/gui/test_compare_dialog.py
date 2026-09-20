"""Comparing any two NIfTI images.

The side-by-side viewer was built for defacing and is not about defacing: raw
against preprocessed, two echoes, a derivative against the scan it came from,
one subject against another. What is tested here is that it is genuinely
general, that two images which do NOT match still open, and that the sync
covers everything the defacing view covers, because it is the same widget.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from bidsmgr.deface.engines import TEMPLATE  # noqa: E402
from bidsmgr.gui.compare_dialog import (  # noqa: E402
    CompareDialog,
    is_nifti,
    open_compare,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def images(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    a = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    b = root / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
    shutil.copyfile(TEMPLATE, a)
    shutil.copyfile(TEMPLATE, b)
    return root, a, b


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Do not read the developer's own remembered view mode.

    The pane opens a scan in the layout the user was last in, which is a real
    preference stored in QSettings. A test that reads it passes or fails
    depending on what the person running it last clicked.
    """
    from PyQt6.QtCore import QSettings

    from bidsmgr.gui.app_settings import AppSettings

    path = tmp_path / "settings.ini"
    monkeypatch.setattr(
        AppSettings, "_settings",
        staticmethod(lambda: QSettings(str(path), QSettings.Format.IniFormat)),
    )
    yield


@pytest.fixture
def opened(qtbot):
    made = []

    def _open(*args, **kwargs):
        dlg = CompareDialog(*args, **kwargs)
        qtbot.addWidget(dlg)
        made.append(dlg)
        return dlg

    yield _open
    for dlg in made:
        dlg.close()


def _wait(qtbot, dlg):
    qtbot.waitUntil(
        lambda: dlg._panes.left.crosshair_voxel() is not None
        and dlg._panes.right.crosshair_voxel() is not None,
        timeout=20_000,
    )


@pytest.mark.parametrize("name,expected", [
    ("a.nii", True), ("a.nii.gz", True), ("A.NII.GZ", True),
    ("a.json", False), ("a.tsv", False), ("a.niftii", False),
])
def test_what_counts_as_an_image(name, expected):
    assert is_nifti(Path(name)) is expected


def test_two_images_open_side_by_side(qtbot, opened, images):
    root, a, b = images
    dlg = opened(a, b, root=root)
    _wait(qtbot, dlg)

    assert dlg._panes.left.current_file() == a
    assert dlg._panes.right.current_file() == b


def test_it_opens_with_nothing_chosen(qtbot, opened, images):
    """A Tools entry has no selection to work from, so it must still open."""
    root, _a, _b = images
    dlg = opened(root=root)

    assert dlg._panes.left.current_file() is None
    assert "nothing chosen" in dlg._left_label.text()
    assert "nothing chosen" in dlg._right_label.text()


def test_one_selected_image_lands_on_the_left(qtbot, opened, images):
    root, a, _b = images
    dlg = opened(a, root=root)
    assert dlg._panes.left.current_file() is None, (
        "it should wait for the second image rather than show half a pair"
    )
    assert a.name in dlg._left_label.text()


def test_captions_are_dataset_relative_when_there_is_a_dataset(
    qtbot, opened, images,
):
    root, a, b = images
    dlg = opened(a, b, root=root)
    assert dlg._left_label.text() == "sub-01/anat/sub-01_T1w.nii.gz"


def test_captions_fall_back_to_the_name_outside_a_dataset(
    qtbot, opened, images, tmp_path,
):
    _root, a, b = images
    stray = tmp_path / "elsewhere.nii.gz"
    shutil.copyfile(TEMPLATE, stray)
    dlg = opened(a, stray, root=None)
    assert dlg._right_label.text() == "elsewhere.nii.gz"


def test_images_of_different_shapes_are_still_linked(qtbot, opened, images,
                                                     tmp_path):
    """Matched by position in the scanner, not by voxel index.

    A cropped image shares no voxel index with its source, and the two are
    still pictures of the same head. Following the crosshair through world
    coordinates is what makes them comparable; refusing to follow it at all
    was giving up on the case the feature is most useful for.
    """
    nib = pytest.importorskip("nibabel")
    import numpy as np

    root, a, _b = images
    # Cropped from the START of the third axis, so the same anatomy sits at
    # a different index in each.
    cropped = root / "sub-01" / "anat" / "sub-01_desc-crop_T1w.nii.gz"
    nib.save(nib.load(str(TEMPLATE)).slicer[:, :, 8:], str(cropped))

    dlg = opened(a, cropped, root=root)
    _wait(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._panes.note.text() != "Loading…", timeout=20_000)

    assert dlg._panes.link.isChecked()
    assert not dlg._panes._same_grid
    assert "Different sizes" in dlg._panes.note.text()
    assert "scanner" in dlg._panes.note.text(), (
        "the note does not say HOW they are matched"
    )

    moved = list(dlg._panes.left.crosshair_voxel())
    moved[2] = max(0, moved[2] - 6)
    dlg._panes.left.set_crosshair_voxel(moved)
    dlg._panes.left._broadcast_crosshair()

    # The indices differ by the crop; the WORLD position is the same.
    left_world = np.asarray(dlg._panes.left.crosshair_world())
    right_world = np.asarray(dlg._panes.right.crosshair_world())
    assert np.allclose(left_world, right_world, atol=1.01), (
        f"the crosshair landed somewhere else: {left_world} vs {right_world}"
    )
    assert dlg._panes.right.crosshair_voxel() != dlg._panes.left.crosshair_voxel(), (
        "the indices happen to match, so this no longer tests world mapping"
    )

    # And the plane still follows too.
    dlg._panes.left._shortcut_orientation(0)
    assert dlg._panes.right.view_state()["orientation"] == 0


def test_the_crosshair_is_linked_when_the_shapes_match(qtbot, opened, images):
    root, a, b = images
    dlg = opened(a, b, root=root)
    _wait(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._panes.link.isEnabled(), timeout=20_000)

    moved = list(dlg._panes.left.crosshair_voxel())
    moved[0] = max(0, moved[0] - 5)
    dlg._panes.left.set_crosshair_voxel(moved)
    dlg._panes.left._broadcast_crosshair()

    assert dlg._panes.right.crosshair_voxel() == moved


def test_only_one_toolbar_while_synced(qtbot, opened, images):
    root, a, b = images
    dlg = opened(a, b, root=root)

    assert not dlg._panes.right._toolbar.isVisibleTo(dlg)
    dlg._panes.link.setChecked(False)
    assert dlg._panes.right._toolbar.isVisibleTo(dlg)


def test_closing_mid_load_stops_both_reads(qtbot, opened, images):
    """A running loader QThread destroyed with its pane aborts the process."""
    root, a, b = images
    dlg = opened(a, b, root=root)
    assert dlg._panes.left._loader is not None or dlg._panes.right._loader is not None

    dlg.close()

    assert dlg._panes.left._loader is None
    assert dlg._panes.right._loader is None


def test_open_compare_takes_whatever_was_selected(qtbot, images):
    root, a, b = images
    dlg = open_compare(None, [a, b], root=root)
    qtbot.addWidget(dlg)
    assert (dlg._left, dlg._right) == (a, b)
    dlg.close()


def test_open_compare_ignores_things_that_are_not_images(qtbot, images):
    root, a, _b = images
    sidecar = root / "dataset_description.json"
    dlg = open_compare(None, [sidecar, a], root=root)
    qtbot.addWidget(dlg)
    assert dlg._left == a and dlg._right is None
    dlg.close()


def test_the_window_fits_the_screen(qtbot, opened, images):
    from PyQt6.QtWidgets import QApplication

    root, a, b = images
    dlg = opened(a, b, root=root)
    screen = dlg.screen() or QApplication.primaryScreen()
    if screen is None:
        pytest.skip("no screen")
    assert dlg.width() <= screen.availableGeometry().width()


def test_the_4d_graph_follows(qtbot, opened, images, tmp_path):
    """The toggle announced the state it was LEAVING.

    `_on_graph_toggled` broadcast before it updated the flag it reports, so
    the other pane was told "graph off" at the moment the user turned it on
    and only caught up at the next interaction. Its controls are shared too:
    two time courses drawn over different neighbourhoods are not comparable.
    """
    nib = pytest.importorskip("nibabel")
    import numpy as np

    root, _a, _b = images
    four_d = root / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz"
    four_d.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(0).random((12, 12, 8, 6)).astype("float32")
    other = root / "sub-01" / "func" / "sub-01_task-y_bold.nii.gz"
    for path in (four_d, other):
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

    dlg = opened(four_d, other, root=root)
    _wait(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._panes.left._graph_btn.isEnabled(), timeout=20_000)

    dlg._panes.left._graph_btn.setChecked(True)
    assert dlg._panes.left.view_state()["graph"] is True
    assert dlg._panes.right.view_state()["graph"] is True, (
        "the other pane was told the state the first one was leaving"
    )

    dlg._panes.left._scope_spin.setValue(3)
    assert dlg._panes.right.view_state()["graph_scope"] == 3

    dlg._panes.left._mark_neighbors_box.setChecked(False)
    assert dlg._panes.right.view_state()["graph_marks"] is False

    dlg._panes.left._vol_slider.setValue(3)
    assert dlg._panes.right.view_state()["volume"] == 3
