"""The viewer opens in the layout you left it in.

A viewer that forgets is one you re-configure every time you open it, and the
thing it forgot back to was a single axial slice, which is nobody's preferred
first look at a volume.

The fallback matters as much as the memory. A remembered 3-D layout cannot be
honoured on a machine without a GPU, or for a file the raycaster cannot take
(an RGB volume with the wrong component count). Falling all the way back to
one slice is what produced the complaint; falling back to the three planes
shows the same thing without the render.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402

from bidsmgr.deface.engines import TEMPLATE  # noqa: E402
from bidsmgr.gui.app_settings import AppSettings  # noqa: E402
from bidsmgr.gui.widgets.nifti_viewer_pane import NiftiViewerPane  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.ini"
    monkeypatch.setattr(
        AppSettings, "_settings",
        staticmethod(lambda: QSettings(str(path), QSettings.Format.IniFormat)),
    )
    yield


def _loaded(qtbot, monkeypatch, *, gpu: bool):
    pane = NiftiViewerPane()
    qtbot.addWidget(pane)
    # The host's real GPU must not decide what this test measures.
    pane._gpu_ok = gpu
    pane.set_file(TEMPLATE, TEMPLATE.parent)
    qtbot.waitUntil(lambda: pane._data is not None, timeout=20_000)
    return pane


def test_the_first_ever_scan_opens_multi_planar_without_a_gpu(qtbot, monkeypatch):
    pane = _loaded(qtbot, monkeypatch, gpu=False)
    assert pane.current_mode() == "multi"
    pane.stop_loading()


def test_the_first_ever_scan_opens_multi_planar_3d_with_one(qtbot, monkeypatch):
    pane = _loaded(qtbot, monkeypatch, gpu=True)
    assert pane.current_mode() in ("combo", "multi")
    pane.stop_loading()


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_the_remembered_layout_is_what_opens(qtbot, monkeypatch, mode):
    AppSettings.remember_nifti_view_mode(mode)
    pane = _loaded(qtbot, monkeypatch, gpu=False)
    assert pane.current_mode() == mode
    pane.stop_loading()


def test_a_remembered_3d_layout_falls_back_to_the_planes_not_one_slice(
    qtbot, monkeypatch,
):
    """The complaint, in one test: it used to land on a single axial slice."""
    AppSettings.remember_nifti_view_mode("combo")
    pane = _loaded(qtbot, monkeypatch, gpu=False)

    assert pane.current_mode() == "multi", (
        "a 3-D layout that cannot be honoured fell back to a single slice"
    )
    pane.stop_loading()


def test_switching_layout_is_remembered_immediately(qtbot, monkeypatch):
    """Written as the user switches: the Editor is not always closed cleanly."""
    pane = _loaded(qtbot, monkeypatch, gpu=False)

    pane._tri_btn.setChecked(False)          # -> single
    assert AppSettings.load().nifti_view_mode == "single"

    pane._tri_btn.setChecked(True)           # -> multi
    assert AppSettings.load().nifti_view_mode == "multi"
    pane.stop_loading()


def test_a_junk_setting_is_ignored_rather_than_obeyed(qtbot, monkeypatch):
    AppSettings._settings().setValue("editor/nifti_view_mode", "sideways")
    assert AppSettings.load().nifti_view_mode == ""

    pane = _loaded(qtbot, monkeypatch, gpu=False)
    assert pane.current_mode() == "multi"
    pane.stop_loading()


def test_later_scans_keep_whatever_view_you_are_in(qtbot, monkeypatch):
    """Only the FIRST volume adopts the preference; then you are driving."""
    pane = _loaded(qtbot, monkeypatch, gpu=False)
    pane._tri_btn.setChecked(False)
    assert pane.current_mode() == "single"

    pane.set_file(None, None)
    pane.set_file(TEMPLATE, TEMPLATE.parent)
    qtbot.waitUntil(lambda: pane._data is not None, timeout=20_000)

    assert pane.current_mode() == "single"
    pane.stop_loading()


# ---------------------------------------------------------------------------
# The layout has to survive looking at something else.
#
# `_clear` drops the pane to a single plane to free the GPU textures, and it
# runs whenever the selection moves to anything that is not a NIfTI. Applying
# the preference only once per session meant the layout held until the first
# time the user clicked a sidecar, and then every scan opened on one slice
# with whatever plane happened to be left over.


def _reload(qtbot, pane):
    """What the Editor does when you click a .json and then a NIfTI again."""
    pane.set_file(None, None)
    pane.set_file(TEMPLATE, TEMPLATE.parent)
    qtbot.waitUntil(lambda: pane._data is not None, timeout=20_000)


def test_clicking_away_and_back_keeps_the_layout(qtbot, monkeypatch):
    pane = _loaded(qtbot, monkeypatch, gpu=False)
    assert pane.current_mode() == "multi"

    _reload(qtbot, pane)

    assert pane.current_mode() == "multi", (
        "the layout was lost the moment the user looked at a sidecar"
    )
    pane.stop_loading()


def test_clicking_away_and_back_keeps_the_plane(qtbot, monkeypatch):
    """The coronal slice people kept landing on was a leftover orientation."""
    pane = _loaded(qtbot, monkeypatch, gpu=False)
    pane._tri_btn.setChecked(False)          # single
    pane._user_set_orientation(0)            # sagittal
    assert pane.current_mode() == "single"

    _reload(qtbot, pane)

    assert pane.current_mode() == "single"
    assert pane.view_state()["orientation"] == 0
    pane.stop_loading()


def test_a_second_pane_opens_the_way_the_first_was_left(qtbot, monkeypatch):
    """A new Editor session, in effect: the preference is on disk."""
    first = _loaded(qtbot, monkeypatch, gpu=False)
    first._tri_btn.setChecked(False)
    first._user_set_orientation(1)           # coronal, deliberately
    first.stop_loading()

    second = _loaded(qtbot, monkeypatch, gpu=False)
    assert second.current_mode() == "single"
    assert second.view_state()["orientation"] == 1
    second.stop_loading()


def test_the_default_plane_is_axial_when_nothing_was_ever_chosen(
    qtbot, monkeypatch,
):
    pane = _loaded(qtbot, monkeypatch, gpu=False)
    assert pane.view_state()["orientation"] == 2
    pane.stop_loading()
