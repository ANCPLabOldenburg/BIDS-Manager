"""Renaming from a right-click offers THAT recording, not the whole subject.

Two reported defects, both about a preview that answered a wider question than
the one that was asked:

* right-clicking one file and choosing "rename subject" ticked every file in
  the dataset carrying that subject, across every datatype. The plan should
  still CONTAIN them, so widening is a click, but the default should be what
  was clicked and what travels with it;
* ticking "merge them into one subject" re-planned, and the re-plan discarded
  whatever had just been selected and ticked everything again.

The second is the worse of the two. A user narrows the selection, ticks merge
to see what a merge would do, and the narrowing silently evaporates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from bidsmgr.gui.rename_entity_dialog import RenameEntityDialog

pytestmark = pytest.mark.gui


def _write(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Two subjects in DIFFERENT sessions, so a merge is actually possible.

    Identical session labels would collide on every file, the merge plan would
    be empty, and a test asserting on its selection would be asserting on
    nothing.
    """
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv", "participant_id\nsub-001\nsub-002\n")
    for sub, ses in (("sub-001", "ses-01"), ("sub-002", "ses-02")):
        _write(root, f"{sub}/{ses}/anat/{sub}_{ses}_T1w.nii.gz")
        _write(root, f"{sub}/{ses}/anat/{sub}_{ses}_T1w.json", "{}")
        _write(root, f"{sub}/{ses}/func/{sub}_{ses}_task-rest_bold.nii.gz")
        _write(root, f"{sub}/{ses}/eeg/{sub}_{ses}_task-rest_eeg.edf")
    return root


@pytest.fixture
def clicked(dataset: Path) -> Path:
    return dataset / "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz"


def _dialog(qtbot, root, focus=None, new="009"):
    dlg = RenameEntityDialog(root, entity="sub", value="001", focus=focus)
    qtbot.addWidget(dlg)
    dlg._new.setText(new)
    dlg.plan_now()
    return dlg


# ---------------------------------------------------------------------------
# What starts ticked


def test_a_clicked_file_ticks_only_itself_and_its_companions(
    qtbot, dataset: Path, clicked: Path,
) -> None:
    dlg = _dialog(qtbot, dataset, focus=clicked)
    assert dlg.selected_keys() == {
        "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz",
        "sub-001/ses-01/anat/sub-001_ses-01_T1w.json",
    }, "the sidecar travels with the recording; the eeg and func do not"


def test_the_plan_still_offers_the_whole_subject(
    qtbot, dataset: Path, clicked: Path,
) -> None:
    """Narrowing the DEFAULT must not narrow what is available, or widening
    would mean closing the dialog and starting again."""
    dlg = _dialog(qtbot, dataset, focus=clicked)
    assert len(dlg._plan.file_moves) > len(dlg.selected_keys())
    assert dlg._preview.total_keys() == len(dlg._plan.file_moves)


def test_with_no_click_everything_is_ticked(qtbot, dataset: Path) -> None:
    """Opened from the toolbar, nothing was pointed at, so the whole subject
    is the only sensible default."""
    dlg = _dialog(qtbot, dataset)
    assert len(dlg.selected_keys()) == len(dlg._plan.file_moves)


def test_clicking_a_folder_ticks_everything_under_it(
    qtbot, dataset: Path,
) -> None:
    dlg = _dialog(qtbot, dataset, focus=dataset / "sub-001/ses-01/eeg")
    assert dlg.selected_keys() == {
        "sub-001/ses-01/eeg/sub-001_ses-01_task-rest_eeg.edf",
    }


def test_the_user_can_still_widen(qtbot, dataset: Path, clicked: Path) -> None:
    dlg = _dialog(qtbot, dataset, focus=clicked)
    dlg._set_all(Qt.CheckState.Checked)
    assert len(dlg.selected_keys()) == len(dlg._plan.file_moves)


# ---------------------------------------------------------------------------
# The merge toggle must not throw the selection away


def test_toggling_merge_keeps_the_focus_selection(
    qtbot, dataset: Path, clicked: Path,
) -> None:
    dlg = _dialog(qtbot, dataset, focus=clicked, new="002")
    before = dlg.selected_keys()
    assert before, "the premise: something is selected"

    dlg._fuse.setChecked(True)

    assert dlg._plan.fusion, "the premise: this is now a merge"
    assert dlg.selected_keys() == before, (
        "re-planning discarded the selection and re-ticked everything"
    )


def test_toggling_merge_keeps_a_WIDENED_selection(
    qtbot, dataset: Path, clicked: Path,
) -> None:
    dlg = _dialog(qtbot, dataset, focus=clicked, new="002")
    dlg._set_all(Qt.CheckState.Checked)
    widened = dlg.selected_keys()

    dlg._fuse.setChecked(True)
    assert dlg.selected_keys() == widened


def test_toggling_merge_keeps_a_NARROWED_selection(
    qtbot, dataset: Path,
) -> None:
    """Opened from the toolbar, so everything starts ticked; the user then
    unticks. That choice must survive too."""
    dlg = _dialog(qtbot, dataset, new="002")
    dlg._set_all(Qt.CheckState.Unchecked)
    one = next(
        item for item in dlg._preview._leaves()
        if item.data(0, Qt.ItemDataRole.UserRole + 1)
    )
    one.setCheckState(0, Qt.CheckState.Checked)
    narrowed = dlg.selected_keys()
    assert len(narrowed) == 1

    dlg._fuse.setChecked(True)
    assert dlg.selected_keys() == narrowed


def test_a_DIFFERENT_rename_gets_a_fresh_default(
    qtbot, dataset: Path, clicked: Path,
) -> None:
    """Remembering the selection must not outlive the operation it belongs to.

    Retyping the target is a different rename, so the default applies again
    rather than a stale set carried over from the last one.
    """
    dlg = _dialog(qtbot, dataset, focus=clicked, new="002")
    dlg._set_all(Qt.CheckState.Checked)
    assert len(dlg.selected_keys()) > 2

    dlg._new.setText("077")
    dlg.plan_now()
    assert len(dlg.selected_keys()) == 2, "back to the clicked file and sidecar"
