"""The comparison view on a machine with no GPU.

Its own file, and that is not tidiness. Creating a pane that pre-builds a
QOpenGLWidget and then one that does not, in the same process, destabilises
Qt under the offscreen platform: the mixed file crashed once in eight runs
while either half alone passed eight out of eight. A machine either has a
GPU or it does not, so the mixture is not a state any user is ever in.

What is checked is that the 3-D half is ABSENT rather than broken. No render
is built, the shared state reports it as missing rather than omitting it, and
a state that arrived from a machine which HAS one is applied without
complaint, because the same dataset gets opened on a workstation and on a
laptop.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402

from bidsmgr.deface.engines import TEMPLATE  # noqa: E402
from bidsmgr.gui.app_settings import AppSettings  # noqa: E402
from bidsmgr.gui.compare_dialog import CompareDialog  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch, tmp_path):
    """No GPU anywhere in this process, and no stored view preference.

    Patched at the SOURCE: the pane pre-creates its GL page in ``__init__``
    when a GPU is present, so a flag flipped on the instance afterwards is
    flipped too late.
    """
    from bidsmgr.gui.widgets import nifti_gl_view

    monkeypatch.setattr(nifti_gl_view, "gpu_available", lambda: False)
    path = tmp_path / "settings.ini"
    monkeypatch.setattr(
        AppSettings, "_settings",
        staticmethod(lambda: QSettings(str(path), QSettings.Format.IniFormat)),
    )
    yield


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


def test_it_opens_and_the_2d_half_syncs_completely(qtbot, opened, images):
    root, a, b = images
    dlg = opened(a, b, root=root)
    _wait(qtbot, dlg)

    assert dlg._panes.left._gl is None, "a 3-D view was built without a GPU"
    state = dlg._panes.left.view_state()
    # Present and None, not absent: a linked pane can then tell "there is no
    # 3-D here" from "this state predates 3-D".
    assert state["camera"] is None
    assert state["clip"] is None
    assert state["gl_controls"] is None

    dlg._panes.left._shortcut_orientation(1)
    assert dlg._panes.right.view_state()["orientation"] == 1

    moved = list(dlg._panes.left.crosshair_voxel())
    moved[0] = max(0, moved[0] - 4)
    dlg._panes.left.set_crosshair_voxel(moved)
    dlg._panes.left._broadcast_crosshair()
    assert dlg._panes.right.crosshair_voxel() == moved


def test_the_first_scan_opens_multi_planar_not_a_single_slice(
    qtbot, opened, images,
):
    root, a, b = images
    dlg = opened(a, b, root=root)
    _wait(qtbot, dlg)
    assert dlg._panes.left.current_mode() == "multi"


def test_a_state_carrying_3d_is_applied_harmlessly(qtbot, opened, images):
    """The same dataset gets opened on a workstation and on a laptop."""
    root, a, b = images
    dlg = opened(a, b, root=root)
    _wait(qtbot, dlg)

    from_a_gpu_machine = dict(dlg._panes.left.view_state())
    from_a_gpu_machine.update({
        "mode": "combo",
        "camera": {"az": 0.4, "el": 0.2, "dist": 2.0, "target": [0, 0, 0]},
        "clip": {"active": 1, "flip": True, "pos": 0.3, "az": 10.0,
                 "el": 5.0, "thick": 0.1},
        "gl_controls": {"_effect": 2},
    })

    dlg._panes.right.apply_view_state(from_a_gpu_machine)

    # Refused up front and dropped to the nearest layout it CAN show, which
    # is the three planes and not a single slice.
    assert dlg._panes.right.current_mode() == "multi"
    assert dlg._panes.right._gl is None


def test_a_remembered_3d_layout_does_not_strand_it_on_one_slice(
    qtbot, opened, images,
):
    AppSettings.remember_nifti_view_mode("combo")
    root, a, b = images
    dlg = opened(a, b, root=root)
    _wait(qtbot, dlg)
    assert dlg._panes.left.current_mode() == "multi"
