"""The entity and session dialog, and the preview tree all three dialogs use.

The preview tree is the larger half of this. It used to group the moves by
their full folder path, which produced ``sub-001/ses-01/anat`` and
``sub-001/ses-01/eeg`` as two unrelated top-level rows that happened to start
with the same text: no subject to collapse, no session to untick, and on a
dataset with more than one of either, no shape at all. These tests pin the
nesting, because a preview that does not look like the dataset is the one part
of a destructive operation the user is supposed to be able to trust.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from bidsmgr.editor import rename as rn
from bidsmgr.gui.edit_entities_dialog import ADD, REMOVE, EditEntitiesDialog

pytestmark = pytest.mark.gui


def _write(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Two subjects, two datatypes each, so nesting has something to nest."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    for sub in ("sub-001", "sub-002"):
        _write(root, f"{sub}/anat/{sub}_T1w.nii.gz")
        _write(root, f"{sub}/anat/{sub}_T1w.json", "{}")
        _write(root, f"{sub}/eeg/{sub}_task-rest_eeg.edf")
        _write(root, f"{sub}/eeg/{sub}_task-rest_eeg.json", "{}")
        _write(root, f"{sub}/{sub}_scans.tsv",
               f"filename\tacq_time\n"
               f"anat/{sub}_T1w.nii.gz\tA\n"
               f"eeg/{sub}_task-rest_eeg.edf\tB\n")
    return root


def _top(dlg: EditEntitiesDialog) -> dict[str, object]:
    tree = dlg._preview
    return {
        tree.topLevelItem(i).text(0): tree.topLevelItem(i)
        for i in range(tree.topLevelItemCount())
    }


def _child_names(item) -> list[str]:
    return [item.child(i).text(0) for i in range(item.childCount())]


def _names(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and ".bidsmgr" not in p.parts
    }


# ---------------------------------------------------------------------------
# The tree looks like the dataset


def test_the_preview_nests_subject_then_datatype(qtbot, dataset: Path) -> None:
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001",
                                       dataset / "sub-002"],
                             session_mode=True)
    qtbot.addWidget(dlg)
    dlg._value.setCurrentText("01")
    dlg.plan_now()

    tops = _top(dlg)
    assert "sub-001" in tops and "sub-002" in tops, (
        "subjects should be top-level rows, not folder paths"
    )
    assert sorted(_child_names(tops["sub-001"])) == ["anat", "eeg"], (
        "datatypes belong UNDER their subject"
    )
    # The defect this replaces: a flat row whose text was the whole path.
    assert not [name for name in tops if "/" in name]


def test_two_subjects_do_not_share_a_datatype_folder(
    qtbot, dataset: Path,
) -> None:
    """Both subjects have an ``anat``. Keying the tree on the folder NAME put
    the second subject's files under the first subject's folder."""
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001",
                                       dataset / "sub-002"],
                             session_mode=True)
    qtbot.addWidget(dlg)
    dlg._value.setCurrentText("01")
    dlg.plan_now()

    tops = _top(dlg)
    for sub in ("sub-001", "sub-002"):
        anat = [tops[sub].child(i) for i in range(tops[sub].childCount())
                if tops[sub].child(i).text(0) == "anat"][0]
        assert all(sub in name for name in _child_names(anat))


def test_a_move_between_folders_shows_the_whole_new_path(
    qtbot, dataset: Path,
) -> None:
    """"Becomes sub-001_ses-01_T1w.nii.gz" would hide the fact that the file
    has moved into a session, which is the larger half of what is about to
    happen."""
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001"], session_mode=True)
    qtbot.addWidget(dlg)
    dlg._value.setCurrentText("01")
    dlg.plan_now()

    tops = _top(dlg)
    anat = [tops["sub-001"].child(i)
            for i in range(tops["sub-001"].childCount())
            if tops["sub-001"].child(i).text(0) == "anat"][0]
    assert anat.child(0).text(1).startswith("sub-001/ses-01/anat/")


def test_things_that_follow_are_shown_but_not_checkable(
    qtbot, dataset: Path,
) -> None:
    """Unticking the row that keeps ``IntendedFor`` pointing at something real
    would produce a half-applied cross-reference."""
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001"], session_mode=True)
    qtbot.addWidget(dlg)
    dlg._value.setCurrentText("01")
    dlg.plan_now()

    follows = _top(dlg).get("Follows automatically")
    assert follows is not None, "the scans table update should be listed"
    assert follows.data(0, Qt.ItemDataRole.CheckStateRole) is None


# ---------------------------------------------------------------------------
# Ticking


def test_unticking_a_subject_drops_everything_under_it(
    qtbot, dataset: Path,
) -> None:
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001",
                                       dataset / "sub-002"],
                             session_mode=True)
    qtbot.addWidget(dlg)
    dlg._value.setCurrentText("01")
    dlg.plan_now()
    total = dlg._preview.total_keys()

    _top(dlg)["sub-002"].setCheckState(0, Qt.CheckState.Unchecked)
    chosen = dlg.selected_keys()

    assert len(chosen) == total // 2
    assert not [key for key in chosen if key.startswith("sub-002")]
    assert dlg._ok.text() == "Apply to selected"


def test_select_none_reaches_every_level(qtbot, dataset: Path) -> None:
    """Qt's auto-tristate only propagates when the change comes from the USER,
    so setting a parent's state in code leaves the children where they were.
    Select none used to leave every file ticked under an unticked folder."""
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001"], session_mode=True)
    qtbot.addWidget(dlg)
    dlg._value.setCurrentText("01")
    dlg.plan_now()

    dlg._set_all(Qt.CheckState.Unchecked)
    assert dlg.selected_keys() == set()
    assert not dlg._ok.isEnabled()

    dlg._set_all(Qt.CheckState.Checked)
    assert len(dlg.selected_keys()) == dlg._preview.total_keys()
    assert dlg._ok.isEnabled()


# ---------------------------------------------------------------------------
# What the dialog offers


def test_the_entity_row_is_hidden_in_session_mode(
    qtbot, dataset: Path,
) -> None:
    """The entity is not a choice there, it is the subject of the dialog."""
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001"], session_mode=True)
    qtbot.addWidget(dlg)
    assert not dlg._entity.isVisible()
    assert [s.key for s in dlg._slots] == ["ses"]


def test_a_mixed_selection_offers_only_what_suits_both(
    qtbot, dataset: Path,
) -> None:
    anat = dataset / "sub-001/anat/sub-001_T1w.nii.gz"
    eeg = dataset / "sub-001/eeg/sub-001_task-rest_eeg.edf"

    one = EditEntitiesDialog(dataset, [anat])
    both = EditEntitiesDialog(dataset, [anat, eeg])
    qtbot.addWidget(one)
    qtbot.addWidget(both)

    alone = {s.key for s in one._slots}
    together = {s.key for s in both._slots}
    assert "echo" in alone and "echo" not in together
    assert "acq" in together


def test_remove_mode_offers_only_optional_entities(
    qtbot, dataset: Path,
) -> None:
    eeg = dataset / "sub-001/eeg/sub-001_task-rest_eeg.edf"
    dlg = EditEntitiesDialog(dataset, [eeg], mode=REMOVE)
    qtbot.addWidget(dlg)
    assert "task" not in {s.key for s in dlg._slots}, (
        "task is required for an EEG recording"
    )


def test_the_value_box_offers_labels_already_in_use(
    qtbot, dataset: Path,
) -> None:
    """A second session is almost always spelled like the first one, and
    typing it again by hand is how it gets typed differently."""
    _write(dataset, "sub-003/ses-pre/anat/sub-003_ses-pre_T1w.nii.gz")
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001"], session_mode=True)
    qtbot.addWidget(dlg)
    assert "pre" in [dlg._value.itemText(i)
                     for i in range(dlg._value.count())]


def test_nothing_to_remove_says_so_rather_than_offering_nothing(
    qtbot, dataset: Path,
) -> None:
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001"],
                             mode=REMOVE, session_mode=True)
    qtbot.addWidget(dlg)
    assert not dlg._slots
    assert "not in a session" in dlg._what.text()
    assert not dlg._ok.isEnabled()


# ---------------------------------------------------------------------------
# End to end


def test_applying_a_partial_session_leaves_the_rest_untouched(
    qtbot, dataset: Path,
) -> None:
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001",
                                       dataset / "sub-002"],
                             session_mode=True)
    qtbot.addWidget(dlg)
    dlg._value.setCurrentText("01")
    dlg.plan_now()
    _top(dlg)["sub-002"].setCheckState(0, Qt.CheckState.Unchecked)

    touched, errors = rn.apply_rename(
        dataset, dlg._plan, only=dlg.selected_keys(),
    )
    assert not errors and touched

    names = _names(dataset)
    assert "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz" in names
    assert "sub-002/anat/sub-002_T1w.nii.gz" in names
    assert (dataset / "sub-001/ses-01/sub-001_ses-01_scans.tsv").is_file()
    assert (dataset / "sub-002/sub-002_scans.tsv").is_file()


def test_the_button_is_dead_until_a_plan_exists(qtbot, dataset: Path) -> None:
    dlg = EditEntitiesDialog(dataset, [dataset / "sub-001"], mode=ADD)
    qtbot.addWidget(dlg)
    assert not dlg._ok.isEnabled()
    dlg._entity.setCurrentIndex(dlg._entity.findData("acq"))
    dlg._value.setCurrentText("fast")
    dlg.plan_now()
    assert dlg._ok.isEnabled()
