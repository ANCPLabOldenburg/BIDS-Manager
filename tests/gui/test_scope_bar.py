"""Where a tool acts: what was picked in the tree, or a part of the dataset.

Several Editor tools used to refuse to open without a tree selection, which
made "take the face off every anatomical" reachable only by selecting them all
first. The scope bar is the one control that answers "act on what" for all of
them, so what is checked here is that it resolves to the paths the engines
already take, and that a selection is still the default when there is one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.gui.widgets.scope_bar import PICKED, WHOLE, ScopeBar

pytestmark = pytest.mark.gui


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """Two subjects; one has sessions, one does not."""
    root = tmp_path / "ds"
    (root / "dataset_description.json").parent.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.10.0"})
    )
    for rel in (
        "sub-001/ses-pre/anat/sub-001_ses-pre_T1w.nii.gz",
        "sub-001/ses-pre/func/sub-001_ses-pre_task-x_bold.nii.gz",
        "sub-001/ses-post/anat/sub-001_ses-post_T1w.nii.gz",
        "sub-002/anat/sub-002_T1w.nii.gz",
        "sub-002/dwi/sub-002_dwi.nii.gz",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    # A folder the tool has no business offering as a datatype.
    (root / ".bidsmgr" / "editor").mkdir(parents=True)
    (root / "sub-001" / "notes").mkdir()
    return root


class TestWhatItOffers:
    def test_the_tree_selection_comes_first_when_there_is_one(
        self, qtbot, dataset,
    ):
        picked = [dataset / "sub-002" / "anat"]
        bar = ScopeBar(dataset, picked)
        qtbot.addWidget(bar)
        assert bar._part.itemData(0) == PICKED
        assert bar.targets() == picked

    def test_with_nothing_picked_it_starts_on_the_dataset(self, qtbot, dataset):
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        assert bar._part.itemData(0) == WHOLE
        assert bar.targets() == [dataset]

    def test_every_subject_and_session_is_offered(self, qtbot, dataset):
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        offered = {bar._part.itemData(i) for i in range(bar._part.count())}
        assert offered == {
            WHOLE, "sub-001", "sub-001/ses-post", "sub-001/ses-pre", "sub-002",
        }

    def test_only_datatypes_the_dataset_has_are_offered(self, qtbot, dataset):
        """The standard's twenty on a dataset holding three is unreadable,
        and a folder named ``notes`` is not a datatype."""
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        offered = {
            bar._datatype.itemData(i) for i in range(bar._datatype.count())
        }
        assert offered == {"", "anat", "func", "dwi"}

    def test_dataset_wide_can_be_refused(self, qtbot, dataset):
        """Delete never offers it: plan_delete refuses the dataset root, so
        offering it would only produce a refusal."""
        bar = ScopeBar(dataset, [], allow_dataset=False)
        qtbot.addWidget(bar)
        offered = {bar._part.itemData(i) for i in range(bar._part.count())}
        assert WHOLE not in offered


class TestWhatItResolvesTo:
    def _select(self, bar, part=None, datatype=None):
        if part is not None:
            bar._part.setCurrentIndex(bar._part.findData(part))
        if datatype is not None:
            bar._datatype.setCurrentIndex(bar._datatype.findData(datatype))

    def test_a_subject_is_its_folder(self, qtbot, dataset):
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        self._select(bar, part="sub-002")
        assert bar.targets() == [dataset / "sub-002"]

    def test_a_session_is_its_folder(self, qtbot, dataset):
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        self._select(bar, part="sub-001/ses-pre")
        assert bar.targets() == [dataset / "sub-001" / "ses-pre"]

    def test_a_datatype_narrows_the_whole_dataset(self, qtbot, dataset):
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        self._select(bar, datatype="anat")
        assert bar.targets() == [
            dataset / "sub-001" / "ses-post" / "anat",
            dataset / "sub-001" / "ses-pre" / "anat",
            dataset / "sub-002" / "anat",
        ]

    def test_a_datatype_narrows_a_subject(self, qtbot, dataset):
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        self._select(bar, part="sub-002", datatype="anat")
        assert bar.targets() == [dataset / "sub-002" / "anat"]

    def test_narrowing_never_widens(self, qtbot, dataset):
        """A picked file survives only if it is in that datatype."""
        picked = [dataset / "sub-002" / "anat" / "sub-002_T1w.nii.gz"]
        bar = ScopeBar(dataset, picked)
        qtbot.addWidget(bar)
        self._select(bar, datatype="anat")
        assert bar.targets() == picked
        self._select(bar, datatype="dwi")
        assert bar.targets() == []

    def test_an_empty_scope_is_reported_rather_than_hidden(
        self, qtbot, dataset,
    ):
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        self._select(bar, part="sub-002", datatype="func")
        assert bar.targets() == []
        assert "Widen" in bar._summary.text()

    def test_changing_either_combo_announces_it(self, qtbot, dataset):
        bar = ScopeBar(dataset, [])
        qtbot.addWidget(bar)
        with qtbot.waitSignal(bar.changed, timeout=500):
            self._select(bar, part="sub-002")
        with qtbot.waitSignal(bar.changed, timeout=500):
            self._select(bar, datatype="anat")
