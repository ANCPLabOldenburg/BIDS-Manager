"""Choosing an image from the dataset, instead of from the file system.

The OS file dialog is the wrong tool for this twice over: it makes the user
navigate folders from memory, and it shows every file when only a handful are
images. Finding `sub-014/ses-post/anat` that way in a sixty-subject dataset
takes longer than the comparison it is standing in the way of.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from bidsmgr.gui.widgets.nifti_picker import (  # noqa: E402
    NiftiPickerDialog,
    find_images,
    is_nifti,
    matches,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "dataset_description.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.11.1"})
    )
    for sub in ("sub-001", "sub-014"):
        for ses in ("ses-pre", "ses-post"):
            anat = root / sub / ses / "anat"
            anat.mkdir(parents=True)
            (anat / f"{sub}_{ses}_T1w.nii.gz").write_bytes(b"\0" * 8)
            (anat / f"{sub}_{ses}_T1w.json").write_text("{}")
            (anat / f"{sub}_{ses}_T2w.nii.gz").write_bytes(b"\0" * 8)
    # A derivative, which is a thing people compare against.
    deriv = root / "derivatives" / "x" / "sub-001" / "anat"
    deriv.mkdir(parents=True)
    (deriv / "sub-001_desc-brain_T1w.nii.gz").write_bytes(b"\0" * 8)
    # Tool state, which must never be offered.
    (root / ".bidsmgr" / "editor").mkdir(parents=True)
    (root / ".bidsmgr" / "editor" / "backup.nii.gz").write_bytes(b"\0" * 8)
    return root


@pytest.mark.parametrize("query,expected", [
    ("", True),
    ("t1", True),
    ("014 t1", True),
    ("t1 014", True),          # order does not matter
    ("SUB-014", True),         # case does not matter
    ("t2", False),
    ("014 t2", False),
])
def test_the_filter_takes_terms_in_any_order(query, expected):
    assert matches("sub-014/ses-post/anat/sub-014_ses-post_T1w.nii.gz",
                   query) is expected


def test_it_finds_images_and_ignores_everything_else(dataset):
    found = find_images(dataset)
    names = {p.name for p in found}

    assert all(is_nifti(p) for p in found)
    assert "sub-001_ses-pre_T1w.json" not in names
    assert "sub-001_desc-brain_T1w.nii.gz" in names, (
        "a derivative is one of the most common things to compare against"
    )
    assert "backup.nii.gz" not in names, "tool state was offered as data"


def test_the_tree_is_shaped_like_the_dataset(qtbot, dataset):
    dlg = NiftiPickerDialog(dataset)
    qtbot.addWidget(dlg)

    tops = {dlg._tree.topLevelItem(i).text(0)
            for i in range(dlg._tree.topLevelItemCount())}
    assert {"sub-001", "sub-014", "derivatives"} <= tops


def test_two_subjects_do_not_share_one_anat_branch(qtbot, dataset):
    """Keying a group on the folder NAME merges them. Keyed on the path."""
    dlg = NiftiPickerDialog(dataset)
    qtbot.addWidget(dlg)

    anats = [item for item in _all_items(dlg._tree) if item.text(0) == "anat"]
    assert len(anats) >= 4, f"only {len(anats)} anat branches for 4 sessions"


def test_filtering_hides_what_does_not_match_and_empty_groups(qtbot, dataset):
    dlg = NiftiPickerDialog(dataset)
    qtbot.addWidget(dlg)

    dlg._filter.setText("014 t2")

    visible = [
        Path(i.data(0, Qt.ItemDataRole.UserRole)).name
        for i in _leaves(dlg._tree) if not i.isHidden()
    ]
    assert visible and all("T2w" in name for name in visible)
    assert all("sub-014" in Path(
        next(i for i in _leaves(dlg._tree)
             if not i.isHidden()).data(0, Qt.ItemDataRole.UserRole)
    ).parts for _ in [0])

    # The other subject's branch is gone, not left empty.
    sub001 = next(
        dlg._tree.topLevelItem(i)
        for i in range(dlg._tree.topLevelItemCount())
        if dlg._tree.topLevelItem(i).text(0) == "sub-001"
    )
    assert sub001.isHidden()


def test_clearing_the_filter_brings_everything_back(qtbot, dataset):
    dlg = NiftiPickerDialog(dataset)
    qtbot.addWidget(dlg)
    total = len(list(_leaves(dlg._tree)))

    dlg._filter.setText("t2")
    assert len([i for i in _leaves(dlg._tree) if not i.isHidden()]) < total

    dlg._filter.setText("")
    assert len([i for i in _leaves(dlg._tree) if not i.isHidden()]) == total


def test_the_count_says_how_much_is_shown(qtbot, dataset):
    dlg = NiftiPickerDialog(dataset)
    qtbot.addWidget(dlg)
    assert "image(s) in this dataset" in dlg._status.text()

    dlg._filter.setText("t2")
    assert "of" in dlg._status.text() and "shown" in dlg._status.text()


def test_nothing_is_chosen_until_something_is_selected(qtbot, dataset):
    dlg = NiftiPickerDialog(dataset)
    qtbot.addWidget(dlg)
    assert dlg.chosen() is None
    assert not dlg._ok.isEnabled()

    leaf = next(_leaves(dlg._tree))
    dlg._tree.setCurrentItem(leaf)
    assert dlg.chosen() is not None
    assert dlg._ok.isEnabled()


def test_a_group_row_is_not_a_choice(qtbot, dataset):
    dlg = NiftiPickerDialog(dataset)
    qtbot.addWidget(dlg)
    group = dlg._tree.topLevelItem(0)

    dlg._tree.setCurrentItem(group)
    assert dlg.chosen() is None
    assert not dlg._ok.isEnabled()


def test_it_opens_on_the_image_it_was_given(qtbot, dataset):
    start = dataset / "sub-014" / "ses-post" / "anat" / "sub-014_ses-post_T2w.nii.gz"
    dlg = NiftiPickerDialog(dataset, start=start)
    qtbot.addWidget(dlg)
    assert dlg.chosen() == start


def test_without_a_dataset_it_says_to_browse(qtbot):
    dlg = NiftiPickerDialog(None)
    qtbot.addWidget(dlg)
    assert "Browse" in dlg._status.text()
    assert not dlg._ok.isEnabled()


def test_an_empty_dataset_says_so_rather_than_showing_nothing(qtbot, tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    dlg = NiftiPickerDialog(root)
    qtbot.addWidget(dlg)
    assert "No NIfTI images" in dlg._status.text()


def _all_items(tree):
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        yield item
        stack.extend(item.child(i) for i in range(item.childCount()))


def _leaves(tree):
    for item in _all_items(tree):
        if item.data(0, Qt.ItemDataRole.UserRole):
            yield item
