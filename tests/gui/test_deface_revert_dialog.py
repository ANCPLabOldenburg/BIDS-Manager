"""The restore dialog, offscreen.

What it has to get right is what it OFFERS. An image defaced during conversion
has no undefaced copy anywhere, by design, and listing it as restorable would
promise something the tool cannot do. Saying why it is absent is the whole
value of opening the dialog on a dataset with nothing to restore.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from bidsmgr.deface import status  # noqa: E402
from bidsmgr.deface.engines import ALLINEATE, TEMPLATE  # noqa: E402
from bidsmgr.gui.deface_revert_dialog import DefaceRevertDialog  # noqa: E402

pytestmark = pytest.mark.gui

REL = "sub-01/anat/sub-01_T1w.nii.gz"
OTHER = "sub-01/anat/sub-01_T2w.nii.gz"


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    return root


def _defaced_with_mirror(root: Path, rel: str = REL) -> None:
    image = root / rel
    image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, image)
    mirror = root / "sourcedata" / rel
    mirror.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, mirror)
    image.write_bytes(b"defaced")
    status.sidecar_for(image).write_text(
        json.dumps(status.record({}, ALLINEATE))
    )


@pytest.fixture
def opened(qtbot):
    made = []

    def _open(root, targets=None):
        dlg = DefaceRevertDialog(root, targets)
        qtbot.addWidget(dlg)
        made.append(dlg)
        return dlg

    yield _open
    for dlg in made:
        dlg.close()


def test_it_lists_what_can_be_restored_and_where_from(opened, tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)

    dlg = opened(root)
    assert dlg._checked() == [REL]
    row = dlg._preview.topLevelItem(0)
    assert "sourcedata" in row.text(1), "the user cannot tell where it comes from"
    assert dlg._ok.isEnabled()


def test_a_convert_time_deface_offers_nothing_and_explains(opened, tmp_path):
    """No undefaced copy ever existed, which is the point of doing it there."""
    root = _dataset(tmp_path)
    image = root / REL
    shutil.copyfile(TEMPLATE, image)
    status.sidecar_for(image).write_text(
        json.dumps(status.record({}, ALLINEATE))
    )

    dlg = opened(root)
    assert dlg._checked() == []
    assert not dlg._ok.isEnabled()
    assert "during conversion" in dlg._status.text()


def test_an_image_we_never_defaced_is_not_offered(opened, tmp_path):
    """A mirror alone is not permission to overwrite somebody's data."""
    root = _dataset(tmp_path)
    shutil.copyfile(TEMPLATE, root / REL)
    mirror = root / "sourcedata" / REL
    mirror.parent.mkdir(parents=True)
    shutil.copyfile(TEMPLATE, mirror)

    assert opened(root)._checked() == []


def test_unticking_narrows_what_would_run(opened, tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    _defaced_with_mirror(root, OTHER)

    dlg = opened(root)
    assert len(dlg._checked()) == 2
    dlg._preview.topLevelItem(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert len(dlg._checked()) == 1


def test_select_none_disables_the_button(opened, tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)

    dlg = opened(root)
    dlg._set_all(Qt.CheckState.Unchecked)
    assert not dlg._ok.isEnabled()
    assert "Nothing selected" in dlg._status.text()


def test_targets_narrow_the_dialog(opened, tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    (root / "sub-02" / "anat").mkdir(parents=True)
    _defaced_with_mirror(root, "sub-02/anat/sub-02_T1w.nii.gz")

    dlg = opened(root, [root / "sub-02"])
    assert dlg._checked() == ["sub-02/anat/sub-02_T1w.nii.gz"]


def test_the_controls_lock_while_it_runs(opened, tmp_path):
    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    dlg = opened(root)

    dlg._set_running(True, 1)
    assert not dlg._preview.isEnabled()
    assert not dlg._ok.isEnabled()
    assert not dlg._stop.isHidden() and not dlg._progress.isHidden()

    dlg._set_running(False)
    assert dlg._preview.isEnabled()
    assert dlg._stop.isHidden()


def test_a_stop_is_reported_as_stopped(opened, tmp_path):
    from bidsmgr.deface.revert import RevertOutcome

    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    dlg = opened(root)
    dlg._set_running(True, 1)

    dlg._on_done(RevertOutcome(cancelled=True))
    assert "Stopped" in dlg._status.text()
    assert dlg._preview.isEnabled()


def test_it_actually_restores_through_the_worker(qtbot, opened, tmp_path,
                                                 monkeypatch):
    """End to end on the GUI path, including the thread."""
    from PyQt6.QtWidgets import QMessageBox

    root = _dataset(tmp_path)
    _defaced_with_mirror(root)
    pristine = (root / "sourcedata" / REL).read_bytes()

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
    )
    dlg = opened(root)
    dlg._on_apply()
    qtbot.waitUntil(lambda: dlg._worker is None, timeout=60_000)

    assert dlg.outcome() is not None and dlg.outcome().ok
    assert (root / REL).read_bytes() == pristine
    doc = json.loads(status.sidecar_for(root / REL).read_text())
    assert status.defaced_by_us(doc) is None
