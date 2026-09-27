"""Entity values across a dataset, and giving an index a consistent width.

``run-1`` and ``run-01`` are both valid: the standard's ``index`` format
accepts either. This is a house style, which is exactly why it is a tool the
user reaches for rather than something the scanner decides.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import values as ev
from bidsmgr.editor.rename import RenameError


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-001/func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.10.0"})
    )
    for run, copies in ((1, 3), (2, 2), (10, 1)):
        for i in range(copies):
            name = f"sub-001_task-x_run-{run}_bold.nii.gz"
            if i:
                name = f"sub-001_task-y{i}_run-{run}_bold.nii.gz"
            (root / "sub-001/func" / name).write_bytes(b"x")
    return root


class TestCounts:
    def test_every_value_with_its_file_count(self, dataset):
        assert ev.counts(dataset, "run") == [("1", 3), ("2", 2), ("10", 1)]

    def test_numbers_sort_numerically(self, dataset):
        """run-2 sits between run-1 and run-10, not after them."""
        assert [v for v, _ in ev.counts(dataset, "run")] == ["1", "2", "10"]

    def test_an_absent_entity_has_no_values(self, dataset):
        assert ev.counts(dataset, "echo") == []


class TestIsIndex:
    def test_run_is_an_index(self):
        assert ev.is_index("run")

    def test_acq_is_a_label(self):
        assert not ev.is_index("acq")

    def test_an_unknown_entity_is_neither(self):
        assert not ev.is_index("notathing")


class TestPad:
    """It writes a value AT a width, in both directions.

    It used to be ``zfill`` alone, so it could only add zeros, and a dataset
    already written as ``run-001`` could not be asked for two digits. Half of
    what a tool called "index widths" claims to do was missing.
    """

    @pytest.mark.parametrize("value,width,expected", [
        ("1", 2, "01"),
        ("1", 3, "001"),
        ("01", 2, "01"),
        # Narrowing. The leading zeros are how wide somebody chose to write
        # the index, not part of it: run-001 and run-01 are the same run.
        ("001", 2, "01"),
        ("0001", 1, "1"),
        ("010", 2, "10"),
        # Significant digits are never dropped: the width is a minimum.
        ("100", 2, "100"),
        # Zero is a legitimate index, and "000".lstrip("0") is empty.
        ("000", 2, "00"),
        ("0", 1, "0"),
        ("rest", 2, "rest"),        # not a number
    ])
    def test_pad(self, value, width, expected):
        assert ev.pad(value, width) == expected


class TestPlanPadding:
    def test_only_the_values_that_change(self, dataset):
        plan = ev.plan_padding(dataset, "run", 2)
        assert [(r.old, r.new) for r in plan] == [("1", "01"), ("2", "02")]

    def test_the_file_counts_come_with_it(self, dataset):
        plan = ev.plan_padding(dataset, "run", 2)
        assert {r.old: r.files for r in plan} == {"1": 3, "2": 2}

    def test_a_label_is_refused(self, dataset):
        with pytest.raises(RenameError, match="label, not an index"):
            ev.plan_padding(dataset, "acq", 2)

    @pytest.mark.parametrize("width", [0, 7, -1])
    def test_a_silly_width_is_refused(self, dataset, width):
        with pytest.raises(RenameError, match="between 1 and 6"):
            ev.plan_padding(dataset, "run", width)

    def test_two_values_colliding_is_refused(self, tmp_path):
        """A dataset holding both run-1 and run-01 is the mess this is for,
        and fusing two runs is not something a padding tool should decide."""
        root = tmp_path / "ds"
        (root / "sub-001/func").mkdir(parents=True)
        for value in ("1", "01"):
            (root / f"sub-001/func/sub-001_task-x_run-{value}_bold.nii.gz"
             ).write_bytes(b"x")
        with pytest.raises(RenameError, match="would both become"):
            ev.plan_padding(root, "run", 2)

    def test_nothing_to_do_is_an_empty_plan(self, dataset):
        assert ev.plan_padding(dataset, "run", 1) == []

    def test_it_narrows_a_dataset_that_is_already_padded(self, tmp_path):
        """The reported bug: ticking 2 digits on a dataset written with 3
        found nothing to change, so the tool only ever worked forwards."""
        root = tmp_path / "ds"
        (root / "sub-001/func").mkdir(parents=True)
        for value in ("001", "002", "010"):
            (root / f"sub-001/func/sub-001_task-x_run-{value}_bold.nii.gz"
             ).write_bytes(b"x")
        plan = ev.plan_padding(root, "run", 2)
        assert [(r.old, r.new) for r in plan] == [
            ("001", "01"), ("002", "02"), ("010", "10"),
        ]

    def test_narrowing_keeps_a_value_that_does_not_fit(self, tmp_path):
        root = tmp_path / "ds"
        (root / "sub-001/func").mkdir(parents=True)
        for value in ("001", "100"):
            (root / f"sub-001/func/sub-001_task-x_run-{value}_bold.nii.gz"
             ).write_bytes(b"x")
        plan = ev.plan_padding(root, "run", 2)
        assert [(r.old, r.new) for r in plan] == [("001", "01")]

    def test_narrowing_onto_an_existing_value_is_refused(self, tmp_path):
        """Two spellings of one run must not be fused by a width change."""
        root = tmp_path / "ds"
        (root / "sub-001/func").mkdir(parents=True)
        for value in ("01", "001"):
            (root / f"sub-001/func/sub-001_task-x_run-{value}_bold.nii.gz"
             ).write_bytes(b"x")
        with pytest.raises(RenameError, match="would both become"):
            ev.plan_padding(root, "run", 2)


class TestEndToEnd:
    def test_padding_moves_the_files_and_the_references(self, dataset):
        """The renames are the real ones, so everything that names a file
        follows it."""
        from bidsmgr.editor import rename as rn

        fmap = dataset / "sub-001/fmap"
        fmap.mkdir()
        sidecar = fmap / "sub-001_phasediff.json"
        sidecar.write_text(json.dumps({
            "IntendedFor": ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"]
        }))

        for repad in ev.plan_padding(dataset, "run", 2):
            rn.apply_rename(
                dataset, rn.plan_rename(dataset, "run", repad.old, repad.new)
            )

        assert (dataset / "sub-001/func/sub-001_task-x_run-01_bold.nii.gz").exists()
        assert json.loads(sidecar.read_text())["IntendedFor"] == [
            "bids::sub-001/func/sub-001_task-x_run-01_bold.nii.gz"
        ]

    def test_describe_says_what_would_happen(self, dataset):
        text = ev.describe_padding("run", ev.plan_padding(dataset, "run", 2))
        assert "run-1 to run-01" in text and "5 file(s)" in text

    def test_describe_says_when_there_is_nothing_to_do(self):
        assert "already has that width" in ev.describe_padding("run", [])


class TestScope:
    """An edit confined to one subject or session."""

    @pytest.fixture()
    def two_subjects(self, tmp_path: Path) -> Path:
        root = tmp_path / "ds"
        for sub in ("sub-001", "sub-002"):
            (root / sub / "ses-pre/func").mkdir(parents=True)
            (root / sub / "ses-post/func").mkdir(parents=True)
            for ses in ("pre", "post"):
                (root / sub / f"ses-{ses}/func"
                 / f"{sub}_ses-{ses}_task-x_run-1_bold.nii.gz").write_bytes(b"x")
        return root

    def test_the_scopes_are_dataset_subject_session(self, two_subjects):
        labels = [s.label for s in ev.scopes(two_subjects)]
        assert labels[0] == "The whole dataset"
        assert "sub-001" in labels
        assert "sub-001 / ses-pre" in labels

    def test_counts_are_confined_to_the_scope(self, two_subjects):
        whole, subject = ev.scopes(two_subjects)[0], ev.scopes(two_subjects)[1]
        assert ev.counts_in(two_subjects, "task", whole) == [("x", 4)]
        assert ev.counts_in(two_subjects, "task", subject) == [("x", 2)]

    def test_a_scoped_replace_touches_only_that_scope(self, two_subjects):
        from bidsmgr.editor import rename as rn

        scope = next(s for s in ev.scopes(two_subjects) if s.label == "sub-001")
        plan, keys = ev.plan_replace(two_subjects, "task", "x", "rest", scope)
        rn.apply_rename(two_subjects, plan, only=keys)

        assert (two_subjects / "sub-001/ses-pre/func"
                / "sub-001_ses-pre_task-rest_run-1_bold.nii.gz").exists()
        assert (two_subjects / "sub-002/ses-pre/func"
                / "sub-002_ses-pre_task-x_run-1_bold.nii.gz").exists(), \
            "the other subject is untouched"

    def test_a_session_scope_is_narrower_still(self, two_subjects):
        from bidsmgr.editor import rename as rn

        scope = next(s for s in ev.scopes(two_subjects)
                     if s.label == "sub-001 / ses-pre")
        plan, keys = ev.plan_replace(two_subjects, "task", "x", "rest", scope)
        rn.apply_rename(two_subjects, plan, only=keys)

        assert (two_subjects / "sub-001/ses-pre/func"
                / "sub-001_ses-pre_task-rest_run-1_bold.nii.gz").exists()
        assert (two_subjects / "sub-001/ses-post/func"
                / "sub-001_ses-post_task-x_run-1_bold.nii.gz").exists(), \
            "the other session is untouched"


class TestInconsistentWidths:
    def test_two_widths_are_found(self, tmp_path):
        root = tmp_path / "ds"
        (root / "sub-001/func").mkdir(parents=True)
        for value in ("1", "01", "2"):
            (root / f"sub-001/func/sub-001_task-x{value}_run-{value}_bold.nii.gz"
             ).write_bytes(b"x")
        splits = ev.inconsistent_widths(root)
        assert len(splits) == 1
        assert splits[0].entity == "run"
        assert splits[0].widths == (1, 2)
        assert splits[0].suggested == 2, "pad up, never lose a digit"

    def test_one_width_is_not_a_finding(self, tmp_path):
        root = tmp_path / "ds"
        (root / "sub-001/func").mkdir(parents=True)
        for value in ("01", "02"):
            (root / f"sub-001/func/sub-001_task-x{value}_run-{value}_bold.nii.gz"
             ).write_bytes(b"x")
        assert ev.inconsistent_widths(root) == []

    def test_a_label_entity_is_never_reported(self, tmp_path):
        """``acq-1`` and ``acq-01`` are two labels, not one at two widths."""
        root = tmp_path / "ds"
        (root / "sub-001/anat").mkdir(parents=True)
        for value in ("1", "01"):
            (root / f"sub-001/anat/sub-001_acq-{value}_T1w.nii.gz").write_bytes(b"x")
        assert [s.entity for s in ev.inconsistent_widths(root)] == []
