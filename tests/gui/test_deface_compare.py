"""The before-and-after dialog, offscreen.

Two things carry the feature. The crosshair link, because comparing two images
at different slices tells you nothing, and the refusal to link images of
different sizes, because a voxel index means a different place in a cropped
image and a silent mismatch would show the user two unrelated slices and let
them conclude the defacer ate the brain.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from bidsmgr.deface import status  # noqa: E402
from bidsmgr.deface.engines import ALLINEATE, TEMPLATE  # noqa: E402
from bidsmgr.gui.deface_compare import DefaceCompareDialog  # noqa: E402
from bidsmgr.project.operations import begin_operation  # noqa: E402

REL = "sub-01/anat/sub-01_T1w.nii.gz"


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    shutil.copyfile(TEMPLATE, root / REL)
    return root


def _defaced(tmp_path: Path) -> Path:
    """A dataset where the image has been defaced, so an original is kept."""
    root = _dataset(tmp_path)
    with begin_operation(root, "Deface 1 image (allineate)") as op:
        op.write_bytes(root / REL, (root / REL).read_bytes())
    sidecar = status.sidecar_for(root / REL)
    sidecar.write_text(json.dumps(status.record({}, ALLINEATE)))
    return root


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
    """Build the dialog and CLOSE it afterwards, however the test ends.

    Closing is not tidiness. Both panes read on their own QThread, parented to
    the pane, so a dialog destroyed while a read is in flight destroys a
    running QThread and Qt aborts the process. The dialog stops its loaders on
    close, and the real caller always closes it because `exec()` returns
    through `done()`. A test that skipped the close would be testing a path
    the application does not have.
    """
    made = []

    def _open(root, rel):
        dlg = DefaceCompareDialog(root, rel)
        qtbot.addWidget(dlg)
        made.append(dlg)
        return dlg

    yield _open
    for dlg in made:
        dlg.close()


def _wait_for_both(qtbot, dlg) -> None:
    qtbot.waitUntil(
        lambda: dlg._before.crosshair_voxel() is not None
        and dlg._after.crosshair_voxel() is not None,
        timeout=20_000,
    )


def test_both_panes_are_bound_to_the_two_versions(qtbot, opened, tmp_path):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)

    assert dlg._after.current_file() == root / REL
    assert dlg._before.current_file() != root / REL
    assert dlg._before.current_file().name == Path(REL).name


def test_moving_the_crosshair_in_one_moves_the_other(qtbot, opened, tmp_path):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._link.isEnabled(), timeout=20_000)

    start = list(dlg._before.crosshair_voxel())
    moved = list(start)
    moved[0] = max(0, start[0] - 5)
    dlg._before.set_crosshair_voxel(moved)
    dlg._before._broadcast_crosshair()

    assert dlg._after.crosshair_voxel() == moved


def test_unlinking_stops_the_mirroring(qtbot, opened, tmp_path):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._link.isEnabled(), timeout=20_000)

    dlg._link.setChecked(False)
    before_other = list(dlg._after.crosshair_voxel())
    moved = list(dlg._before.crosshair_voxel())
    moved[0] = max(0, moved[0] - 7)
    dlg._before.set_crosshair_voxel(moved)
    dlg._before._broadcast_crosshair()

    assert dlg._after.crosshair_voxel() == before_other


def test_different_shapes_still_link_and_say_how(qtbot, opened, tmp_path):
    """The neck-cropping engine legitimately produces a smaller image."""
    nib = pytest.importorskip("nibabel")
    root = _defaced(tmp_path)
    img = nib.load(str(TEMPLATE))
    cropped = img.slicer[:, :, 5:]
    nib.save(cropped, str(root / REL))

    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._link_note.text() != "Loading…", timeout=20_000)

    # Sync stays ON, crosshair included: it travels as millimetres, so a
    # cropped result still points at the same anatomy as its source.
    import numpy as np

    assert dlg._link.isChecked()
    assert not dlg._same_grid
    assert "Different sizes" in dlg._link_note.text()
    assert "scanner" in dlg._link_note.text()

    moved = list(dlg._before.crosshair_voxel())
    moved[2] = max(0, moved[2] - 4)
    dlg._before.set_crosshair_voxel(moved)
    dlg._before._broadcast_crosshair()

    assert np.allclose(
        np.asarray(dlg._before.crosshair_world()),
        np.asarray(dlg._after.crosshair_world()),
        atol=1.01,
    ), "the cropped image did not follow to the same place"


def test_closing_mid_load_stops_both_reads(qtbot, opened, tmp_path):
    """The contract the fixture depends on, and the process's life with it.

    Closing a comparison window a moment after opening it is the ordinary
    thing to do. If the reads were still running when the dialog went, Qt
    would abort the whole application, not raise.
    """
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    assert dlg._before._loader is not None or dlg._after._loader is not None

    dlg.close()

    assert dlg._before._loader is None
    assert dlg._after._loader is None


def test_no_original_explains_rather_than_showing_an_empty_pane(
    qtbot, opened, tmp_path
):
    root = _dataset(tmp_path)
    status.sidecar_for(root / REL).write_text(
        json.dumps(status.record({}, ALLINEATE))
    )

    dlg = opened(root, REL)
    assert not hasattr(dlg, "_before")
    assert "No undefaced copy" in dlg._header.text()
    assert "during conversion" in dlg._subhead.text()


def test_the_header_names_the_engine_and_where_the_copy_came_from(
    qtbot, opened, tmp_path
):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)

    assert REL in dlg._header.text()
    text = dlg._subhead.text()
    assert ALLINEATE.label in text
    assert "edit history" in text or "you ran" in text


# ---------------------------------------------------------------------------
# One set of controls, two images.


def test_only_one_toolbar_is_shown_while_the_views_are_synced(
    qtbot, opened, tmp_path,
):
    """Two identical toolbars driving one state is the same control twice."""
    root = _defaced(tmp_path)
    dlg = opened(root, REL)

    # `isVisibleTo`, not `isHidden`: the toolbar lives inside a scroll area
    # (so a narrow pane can shrink) and it is the WRAPPER that gets hidden.
    assert dlg._link.isChecked()
    assert not dlg._after._toolbar.isVisibleTo(dlg)

    dlg._link.setChecked(False)
    assert dlg._after._toolbar.isVisibleTo(dlg), (
        "unsyncing did not give the right image its own controls"
    )

    dlg._link.setChecked(True)
    assert not dlg._after._toolbar.isVisibleTo(dlg)


def test_changing_the_view_in_one_changes_the_other(qtbot, opened, tmp_path):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    # A loaded pane lands in multi-planar, so "single" is the mode that
    # actually proves something moved.
    # A loaded pane lands in the multi-planar layout, or the multi-planar
    # 3-D one where there is a GPU. Either way "single" is a mode it is NOT
    # in, which is what makes the assertion mean something.
    assert dlg._after.current_mode() in ("multi", "combo")

    dlg._before._set_view_mode("single")
    dlg._before._broadcast_view()
    assert dlg._after.current_mode() == "single"

    dlg._before._set_orientation(0)
    dlg._before._broadcast_view()
    assert dlg._after.view_state()["orientation"] == 0


def test_unsyncing_lets_them_differ(qtbot, opened, tmp_path):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    started = dlg._after.current_mode()
    dlg._link.setChecked(False)

    dlg._before._set_view_mode("single" if started != "single" else "multi")
    dlg._before._broadcast_view()
    assert dlg._after.current_mode() == started, (
        "the right pane followed even though sync is off"
    )


def test_re_syncing_adopts_the_left_view(qtbot, opened, tmp_path):
    """Or the two stay however they drifted apart, which is not a comparison."""
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)

    started = dlg._after.current_mode()
    wanted = "single" if started != "single" else "multi"
    dlg._link.setChecked(False)
    dlg._before._set_view_mode(wanted)
    dlg._before._broadcast_view()
    assert dlg._after.current_mode() == started

    dlg._link.setChecked(True)
    assert dlg._after.current_mode() == wanted


def test_the_view_follows_even_when_the_shapes_differ(qtbot, opened, tmp_path):
    """Only the crosshair is meaningless across shapes; the mode is not."""
    nib = pytest.importorskip("nibabel")
    root = _defaced(tmp_path)
    nib.save(nib.load(str(TEMPLATE)).slicer[:, :, 5:], str(root / REL))

    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    qtbot.waitUntil(lambda: dlg._link_note.text() != "Loading…", timeout=20_000)

    started = dlg._after.current_mode()
    wanted = "single" if started != "single" else "multi"
    dlg._before._set_view_mode(wanted)
    dlg._before._broadcast_view()
    assert dlg._after.current_mode() == wanted


def test_the_restore_button_is_offered_when_there_is_an_original(
    qtbot, opened, tmp_path,
):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    assert dlg._restore.isEnabled()


def test_no_original_means_no_restore_button_at_all(qtbot, opened, tmp_path):
    root = _dataset(tmp_path)
    status.sidecar_for(root / REL).write_text(
        json.dumps(status.record({}, ALLINEATE))
    )
    dlg = opened(root, REL)
    assert not hasattr(dlg, "_restore")


# ---------------------------------------------------------------------------
# The 3-D half of sync, and the window fitting on the screen.


def test_the_pane_can_be_made_narrow(qtbot, opened, tmp_path):
    """Two viewer panes used to pin the dialog at ~1750 px and grow it.

    A viewer pane's natural minimum is its widest row of controls, which is
    fine for the single pane in the Editor and not fine for two of them: the
    window could not be dragged narrower, and it widened itself the moment
    the images finished loading.
    """
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)

    assert dlg._before.minimumSizeHint().width() < 420
    assert dlg._after.minimumSizeHint().width() < 420


def test_the_window_is_not_opened_bigger_than_the_screen(qtbot, opened, tmp_path):
    from PyQt6.QtWidgets import QApplication

    dlg = opened(_defaced(tmp_path), REL)
    screen = dlg.screen() or QApplication.primaryScreen()
    if screen is None:
        pytest.skip("no screen")
    available = screen.availableGeometry()
    assert dlg.width() <= available.width()
    assert dlg.height() <= available.height()


def test_the_camera_is_part_of_the_shared_view_state(qtbot, opened, tmp_path):
    """Rotating one head and not the other is not a comparison."""
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)

    state = dlg._before.view_state()
    assert "camera" in state and "gl_controls" in state
    # Without a GPU the 3-D view is never built, and both read None rather
    # than being absent, so a linked pane has nothing to apply.
    if state["camera"] is None:
        pytest.skip("no 3-D view on this host")

    moved = dict(state["camera"])
    moved["az"] = moved["az"] + 0.5
    dlg._before._gl.apply_camera_state(moved)
    dlg._before._broadcast_view()

    assert dlg._after._gl is not None
    assert abs(dlg._after._gl.camera_state()["az"] - moved["az"]) < 1e-6


def test_moving_the_camera_tells_the_other_pane(qtbot, opened, tmp_path):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    if dlg._before.view_state()["camera"] is None:
        pytest.skip("no 3-D view on this host")

    seen: list = []
    dlg._before.view_changed.connect(seen.append)
    dlg._before._gl.camera_changed.emit()
    assert seen, "a camera move did not reach the pane's view_changed signal"


def test_an_effect_change_reaches_the_other_render(qtbot, opened, tmp_path):
    """Two renders showing the same brain under different effects compare
    nothing. Effects, lighting, thresholds and the clip plane all follow."""
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    if dlg._before.view_state()["gl_controls"] is None:
        pytest.skip("no 3-D view on this host")

    controls = dlg._before._gl_controls
    before = controls.controls_state()
    key = "_effect"
    assert key in before, "the effect selector is not part of the shared state"
    controls._effect.setCurrentIndex(
        (before[key] + 1) % controls._effect.count()
    )

    assert dlg._after._gl_controls is not None
    assert (
        dlg._after._gl_controls.controls_state()[key]
        == controls.controls_state()[key]
    ), "the second render kept the old effect"


# ---------------------------------------------------------------------------
# Sync has to happen WHEN it happens, not at the next interaction.
#
# Both of these worked in the sense that the state was shared, and were
# useless in practice: the second pane caught up only when something else
# touched it, so switching to coronal showed one coronal image beside one
# axial image until you clicked.


def test_pressing_an_orientation_pill_switches_both(qtbot, opened, tmp_path):
    """Through the BUTTON, not the setter underneath it.

    The setter also runs while the pane is being built and while a file is
    being bound, so the announcement lives on the user's entry point. A test
    that called the setter directly would pass against the broken version.
    """
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    # The pills only apply to a single plane, so they are disabled in the
    # multi-planar (or multi-planar 3-D) view a loaded pane lands in. This is
    # the state a user is in when they press one.
    dlg._before._set_view_mode("single")
    assert dlg._before._sa_btn.isEnabled()

    dlg._before._sa_btn.click()
    assert dlg._before.view_state()["orientation"] == 0
    assert dlg._after.view_state()["orientation"] == 0, (
        "the right image kept the old plane until the next interaction"
    )

    dlg._before._co_btn.click()
    assert dlg._after.view_state()["orientation"] == 1


def test_the_keyboard_shortcut_switches_both(qtbot, opened, tmp_path):
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)

    dlg._before._shortcut_orientation(0)
    assert dlg._after.view_state()["orientation"] == 0


def test_moving_the_clip_plane_reaches_the_other_render(qtbot, opened, tmp_path):
    """Slicing in 3-D is driven from inside the GL widget.

    It mirrors itself into the control sliders silently, so without listening
    for `clip_changed` the second render only caught up on the next click.
    """
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    if dlg._before.view_state()["gl_controls"] is None:
        pytest.skip("no 3-D view on this host")

    seen: list = []
    dlg._before.view_changed.connect(seen.append)
    dlg._before._gl.clip_changed.emit()
    assert seen, "a clip-plane change never reached the pane's view_changed"


def test_inverting_the_cut_inverts_it_in_both(qtbot, opened, tmp_path):
    """Shift+X flips which side of the plane is kept.

    It has no slider, so a sync built from the control widgets alone shared
    the plane's angle and depth and left the two renders cut from OPPOSITE
    sides, which is the one difference that makes a comparison actively
    misleading rather than merely unsynchronised.
    """
    root = _defaced(tmp_path)
    dlg = opened(root, REL)
    _wait_for_both(qtbot, dlg)
    if dlg._before.view_state()["clip"] is None:
        pytest.skip("no 3-D view on this host")

    before = dlg._before._gl.clip_state()["flip"]
    dlg._before._gl_controls.kbd_invert()          # Shift+X

    assert dlg._before._gl.clip_state()["flip"] is not before
    assert dlg._after._gl.clip_state()["flip"] == (
        dlg._before._gl.clip_state()["flip"]
    ), "the second render is cut from the other side"
    assert dlg._after._gl.clip_state()["active"] == 1
