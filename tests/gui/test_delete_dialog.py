"""The delete dialog, and the repairs it must not let you untick.

Deletion is the one operation here where a surprise is expensive, so the
dialog's job is to say what goes BEFORE anything does. These tests are mostly
about that promise: the preview is complete, the scale is visible, the repairs
are shown but not choosable, and what is applied is what was ticked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from bidsmgr.editor import remove as rm
from bidsmgr.gui.delete_dialog import DeleteDialog

pytestmark = pytest.mark.gui


def _write(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv",
           "participant_id\tage\nsub-001\t31\nsub-002\t28\n")
    for sub in ("sub-001", "sub-002"):
        _write(root, f"{sub}/ses-01/anat/{sub}_ses-01_T1w.nii.gz")
        _write(root, f"{sub}/ses-01/anat/{sub}_ses-01_T1w.json", "{}")
        _write(root, f"{sub}/ses-01/func/{sub}_ses-01_task-rest_bold.nii.gz")
        _write(root, f"{sub}/ses-01/func/{sub}_ses-01_task-rest_bold.json", "{}")
        _write(root, f"{sub}/ses-01/{sub}_ses-01_scans.tsv",
               "filename\tacq_time\n"
               f"anat/{sub}_ses-01_T1w.nii.gz\tA\n"
               f"func/{sub}_ses-01_task-rest_bold.nii.gz\tB\n")
    return root


def _tops(dlg: DeleteDialog) -> dict:
    tree = dlg._preview
    return {
        tree.topLevelItem(i).text(0): tree.topLevelItem(i)
        for i in range(tree.topLevelItemCount())
    }


# ---------------------------------------------------------------------------
# The preview


def test_the_preview_nests_like_the_dataset(qtbot, dataset: Path) -> None:
    dlg = DeleteDialog(dataset, [dataset / "sub-001/ses-01"])
    qtbot.addWidget(dlg)
    tops = _tops(dlg)
    assert "sub-001" in tops
    assert not [name for name in tops if "/" in name], (
        "a folder path as a row label is the flat grouping this replaced"
    )
    session = tops["sub-001"].child(0)
    assert session.text(0) == "ses-01"
    assert {session.child(i).text(0) for i in range(session.childCount())} >= {
        "anat", "func",
    }


def test_every_leaf_says_what_happens(qtbot, dataset: Path) -> None:
    """The second column is the whole point of the preview."""
    from bidsmgr.gui.widgets.move_preview import KEY_ROLE

    dlg = DeleteDialog(dataset, [dataset / "sub-001/ses-01/func"])
    qtbot.addWidget(dlg)
    leaves = [
        item for item in dlg._preview._leaves()
        if item.data(0, KEY_ROLE)
    ]
    assert leaves, "expected file rows"
    assert all(item.text(1) == "deleted" for item in leaves)


def test_the_repairs_are_shown_but_not_choosable(qtbot, dataset: Path) -> None:
    """Letting somebody untick "remove the scans row" would reintroduce
    exactly the damage this feature exists to prevent."""
    dlg = DeleteDialog(dataset, [dataset / "sub-001"])
    qtbot.addWidget(dlg)
    follows = _tops(dlg).get("Follows automatically")
    assert follows is not None
    assert follows.data(0, Qt.ItemDataRole.CheckStateRole) is None
    rows = [follows.child(i).text(0) for i in range(follows.childCount())]
    assert "participants.tsv" in rows, (
        "deleting a whole subject drops its participants row, and the dialog "
        "has to say so"
    )


def test_the_scale_of_the_deletion_is_visible(qtbot, dataset: Path) -> None:
    """A file count and a size, because "delete this folder" means different
    things at 3 files and at 30 GB."""
    dlg = DeleteDialog(dataset, [dataset / "sub-001/ses-01"])
    qtbot.addWidget(dlg)
    assert "file(s)" in dlg._scope.text()
    assert any(unit in dlg._scope.text() for unit in ("B", "KB", "MB", "GB"))


def test_a_refusal_is_shown_and_the_button_stays_dead(
    qtbot, dataset: Path,
) -> None:
    dlg = DeleteDialog(dataset, [dataset / "dataset_description.json"])
    qtbot.addWidget(dlg)
    assert not dlg._ok.isEnabled()
    assert "Refused" in _tops(dlg)


# ---------------------------------------------------------------------------
# Ticking, and applying what was ticked


def test_the_button_counts_what_is_ticked(qtbot, dataset: Path) -> None:
    dlg = DeleteDialog(dataset, [dataset / "sub-001/ses-01"])
    qtbot.addWidget(dlg)
    total = dlg._preview.total_keys()
    assert dlg._ok.text() == f"Delete {total} file(s)"

    anat = _tops(dlg)["sub-001"].child(0).child(0)
    assert anat.text(0) == "anat"
    anat.setCheckState(0, Qt.CheckState.Unchecked)
    assert len(dlg.selected_keys()) < total
    assert dlg._ok.text() == f"Delete {len(dlg.selected_keys())} file(s)"


def test_select_none_disables_the_button(qtbot, dataset: Path) -> None:
    dlg = DeleteDialog(dataset, [dataset / "sub-001/ses-01"])
    qtbot.addWidget(dlg)
    dlg._set_all(Qt.CheckState.Unchecked)
    assert dlg.selected_keys() == set()
    assert not dlg._ok.isEnabled()

    dlg._set_all(Qt.CheckState.Checked)
    assert dlg._ok.isEnabled()


def test_a_ticked_scans_table_is_spared_when_a_recording_survives(
    qtbot, dataset: Path,
) -> None:
    """The table is a leaf in the preview like any other file, so unticking a
    recording while leaving the table ticked would delete the thing that
    describes what survived.

    That cannot be left to the user to spot, so a table is taken back out of
    the deletion whenever anything it describes is staying, and edited
    instead.
    """
    dlg = DeleteDialog(dataset, [dataset / "sub-001/ses-01"])
    qtbot.addWidget(dlg)
    table = dataset / "sub-001/ses-01/sub-001_ses-01_scans.tsv"
    assert table in dlg._plan.files, (
        "deleting the whole session should include its table"
    )

    anat = _tops(dlg)["sub-001"].child(0).child(0)
    assert anat.text(0) == "anat"
    anat.setCheckState(0, Qt.CheckState.Unchecked)
    rm.apply_delete(dataset, dlg._plan, only=dlg.selected_keys())

    assert table.is_file(), "the table still describes the surviving anat"


def test_applying_the_ticked_subset_repairs_only_for_it(
    qtbot, dataset: Path,
) -> None:
    """Driven the way the dialog drives it, end to end."""
    dlg = DeleteDialog(dataset, [dataset / "sub-001/ses-01"])
    qtbot.addWidget(dlg)
    anat = _tops(dlg)["sub-001"].child(0).child(0)
    anat.setCheckState(0, Qt.CheckState.Unchecked)

    touched, errors = rm.apply_delete(
        dataset, dlg._plan, only=dlg.selected_keys(),
    )
    assert not errors and touched

    assert (dataset / "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz").is_file()
    assert not (dataset / "sub-001/ses-01/func").exists()

    table = dataset / "sub-001/ses-01/sub-001_ses-01_scans.tsv"
    rows = table.read_text()
    assert "anat/sub-001_ses-01_T1w.nii.gz" in rows, (
        "the row for the file that stayed must not have been removed"
    )
    assert "task-rest_bold" not in rows
    assert "sub-001" in (dataset / "participants.tsv").read_text(), (
        "the subject still has data, so it is still a participant"
    )
