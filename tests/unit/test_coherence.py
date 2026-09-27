"""Whether the dataset's files still agree with each other.

Every case below is legal BIDS and wrong, which is the whole point: the
validator will not report any of them. They are what a dataset looks like
after somebody renamed a folder in Finder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import coherence
from bidsmgr.editor.coherence import Kind


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-001/func").mkdir(parents=True)
    (root / "sub-001/anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.10.0"})
    )
    (root / "participants.tsv").write_text(
        "participant_id\tage\nsub-001\t30\n"
    )
    for rel in ("sub-001/func/sub-001_task-x_bold.nii.gz",
                "sub-001/anat/sub-001_T1w.nii.gz"):
        (root / rel).write_bytes(b"x")
        (root / rel).with_name(
            Path(rel).name.split(".")[0] + ".json"
        ).write_text("{}")
    (root / "sub-001/sub-001_scans.tsv").write_text(
        "filename\tacq_time\n"
        "func/sub-001_task-x_bold.nii.gz\t2026-01-01T09:00:00\n"
        "anat/sub-001_T1w.nii.gz\t2026-01-01T09:05:00\n"
    )
    return root


def _kinds(findings) -> set:
    return {f.kind for f in findings}


def test_a_coherent_dataset_reports_nothing(dataset):
    assert coherence.check(dataset) == []


class TestScansTables:
    def test_a_row_naming_a_deleted_file(self, dataset):
        (dataset / "sub-001/func/sub-001_task-x_bold.nii.gz").unlink()
        (dataset / "sub-001/func/sub-001_task-x_bold.json").unlink()
        found = coherence.check(dataset)
        assert Kind.SCANS_ROW_ORPHAN in _kinds(found)

    def test_repairing_it_drops_only_that_row(self, dataset):
        (dataset / "sub-001/func/sub-001_task-x_bold.nii.gz").unlink()
        (dataset / "sub-001/func/sub-001_task-x_bold.json").unlink()
        coherence.apply(dataset, coherence.check(dataset))
        rows = (dataset / "sub-001/sub-001_scans.tsv").read_text().splitlines()
        assert len(rows) == 2, "header plus the surviving anat row"
        assert "T1w" in rows[1]

    def test_a_recording_with_no_row(self, dataset):
        new = dataset / "sub-001/func/sub-001_task-y_bold.nii.gz"
        new.write_bytes(b"x")
        found = coherence.check(dataset)
        assert Kind.SCANS_ROW_MISSING in _kinds(found)

    def test_repairing_it_adds_the_row(self, dataset):
        new = dataset / "sub-001/func/sub-001_task-y_bold.nii.gz"
        new.write_bytes(b"x")
        coherence.apply(dataset, coherence.check(dataset))
        text = (dataset / "sub-001/sub-001_scans.tsv").read_text()
        assert "func/sub-001_task-y_bold.nii.gz" in text
        assert coherence.check(dataset) == []


class TestParticipants:
    def test_a_row_for_a_deleted_subject(self, dataset):
        (dataset / "participants.tsv").write_text(
            "participant_id\tage\nsub-001\t30\nsub-999\t40\n"
        )
        assert Kind.PARTICIPANT_ORPHAN in _kinds(coherence.check(dataset))

    def test_repairing_it_keeps_the_real_one(self, dataset):
        (dataset / "participants.tsv").write_text(
            "participant_id\tage\nsub-001\t30\nsub-999\t40\n"
        )
        coherence.apply(dataset, coherence.check(dataset))
        text = (dataset / "participants.tsv").read_text()
        assert "sub-001" in text and "sub-999" not in text

    def test_a_subject_with_no_row(self, dataset):
        (dataset / "sub-002/anat").mkdir(parents=True)
        (dataset / "sub-002/anat/sub-002_T1w.nii.gz").write_bytes(b"x")
        assert Kind.PARTICIPANT_MISSING in _kinds(coherence.check(dataset))

    def test_repairing_it_adds_the_row(self, dataset):
        (dataset / "sub-002/anat").mkdir(parents=True)
        (dataset / "sub-002/anat/sub-002_T1w.nii.gz").write_bytes(b"x")
        coherence.apply(dataset, coherence.check(dataset))
        assert "sub-002" in (dataset / "participants.tsv").read_text()


class TestLinks:
    def test_a_pointer_at_nothing(self, dataset):
        sidecar = dataset / "sub-001/anat/sub-001_T1w.json"
        sidecar.write_text(json.dumps({"Sources": ["bids::gone.nii.gz"]}))
        assert Kind.LINK_BROKEN in _kinds(coherence.check(dataset))

    def test_repairing_it_removes_the_key_when_nothing_is_left(self, dataset):
        sidecar = dataset / "sub-001/anat/sub-001_T1w.json"
        sidecar.write_text(json.dumps({"Sources": ["bids::gone.nii.gz"]}))
        coherence.apply(dataset, coherence.check(dataset))
        assert "Sources" not in json.loads(sidecar.read_text())

    def test_a_surviving_entry_is_kept(self, dataset):
        sidecar = dataset / "sub-001/anat/sub-001_T1w.json"
        good = "bids::sub-001/func/sub-001_task-x_bold.nii.gz"
        sidecar.write_text(json.dumps({"Sources": [good, "bids::gone.nii.gz"]}))
        coherence.apply(dataset, coherence.check(dataset))
        assert json.loads(sidecar.read_text())["Sources"] == [good]


class TestOrphanSidecars:
    def test_a_sidecar_with_no_recording(self, dataset):
        (dataset / "sub-001/anat/sub-001_acq-ghost_T1w.json").write_text("{}")
        assert Kind.SIDECAR_ORPHAN in _kinds(coherence.check(dataset))

    def test_an_inherited_sidecar_is_not_one(self, dataset):
        """A sidecar above the datatype folder describes everything below it,
        so it has no single partner to be missing."""
        (dataset / "task-x_bold.json").write_text("{}")
        assert Kind.SIDECAR_ORPHAN not in _kinds(coherence.check(dataset))

    def test_repairing_it_deletes_the_sidecar(self, dataset):
        ghost = dataset / "sub-001/anat/sub-001_acq-ghost_T1w.json"
        ghost.write_text("{}")
        coherence.apply(dataset, coherence.check(dataset))
        assert not ghost.exists()


class TestSafety:
    def test_check_writes_nothing(self, dataset):
        before = {
            p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()
        }
        coherence.check(dataset)
        after = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        assert before == after

    def test_the_tools_own_backups_are_not_checked(self, dataset):
        """An operation copies every file it changes under .bidsmgr/, and
        those copies are scans tables too. Walking into them reported the
        BACKUP's rows as orphans: a pass that flags its own undo history is
        worse than useless."""
        (dataset / "participants.tsv").write_text(
            "participant_id\tage\nsub-001\t30\nsub-999\t40\n"
        )
        coherence.apply(dataset, coherence.check(dataset))
        assert (dataset / ".bidsmgr").exists(), "an original was kept"
        assert coherence.check(dataset) == []

    def test_a_whole_repair_pass_is_one_undo(self, dataset):
        from bidsmgr.project.operations import read_log

        (dataset / "participants.tsv").write_text(
            "participant_id\tage\nsub-001\t30\nsub-999\t40\n"
        )
        (dataset / "sub-001/anat/sub-001_acq-ghost_T1w.json").write_text("{}")
        found = coherence.check(dataset)
        assert len(found) >= 2
        coherence.apply(dataset, found)
        assert len(read_log(dataset)) == 1

    def test_one_failing_check_does_not_stop_the_others(self, dataset, monkeypatch):
        def boom(_root):
            raise RuntimeError("malformed table")

        monkeypatch.setattr(coherence, "CHECKS", (boom, coherence.check_participants))
        (dataset / "participants.tsv").write_text(
            "participant_id\tage\nsub-001\t30\nsub-999\t40\n"
        )
        assert Kind.PARTICIPANT_ORPHAN in _kinds(coherence.check(dataset))


class TestNoScansTableAtAll:
    """``*_scans.tsv`` is RECOMMENDED, not required."""

    def test_a_dataset_that_uses_none_is_coherent(self, dataset):
        (dataset / "sub-001/sub-001_scans.tsv").unlink()
        assert Kind.SCANS_ROW_MISSING not in _kinds(coherence.check(dataset))

    def test_but_a_table_that_exists_must_be_complete(self, dataset):
        (dataset / "sub-001/func/sub-001_task-y_bold.nii.gz").write_bytes(b"x")
        assert Kind.SCANS_ROW_MISSING in _kinds(coherence.check(dataset))


class TestTheNewChecks:
    """Five kinds of disagreement the first version was silent about."""

    def test_an_index_written_at_two_widths(self, dataset):
        f = dataset / "sub-001/func/sub-001_task-x_bold.nii.gz"
        f.rename(f.with_name("sub-001_task-x_run-1_bold.nii.gz"))
        (dataset / "sub-001/func/sub-001_task-x_run-02_bold.nii.gz").write_bytes(b"x")
        found = [f for f in coherence.check(dataset)
                 if f.kind is Kind.ENTITY_WIDTH_SPLIT]
        assert found
        assert "run-1" in found[0].detail and "run-02" in found[0].detail

    def test_the_width_repair_shows_the_exact_renames(self, dataset):
        f = dataset / "sub-001/func/sub-001_task-x_bold.nii.gz"
        f.rename(f.with_name("sub-001_task-x_run-1_bold.nii.gz"))
        (dataset / "sub-001/func/sub-001_task-x_run-02_bold.nii.gz").write_bytes(b"x")
        found = next(f for f in coherence.check(dataset)
                     if f.kind is Kind.ENTITY_WIDTH_SPLIT)
        assert ("run-1", "run-01", 1) in found.changes

    def test_a_clash_padding_cannot_settle_says_why(self, dataset):
        """run-1 beside run-01 cannot be padded: it would fuse two runs.
        A finding that says "no repair" without saying why is one nobody
        can act on."""
        for value in ("1", "01"):
            (dataset / f"sub-001/func/sub-001_task-y_run-{value}_bold.nii.gz"
             ).write_bytes(b"x")
        found = next(f for f in coherence.check(dataset)
                     if f.kind is Kind.ENTITY_WIDTH_SPLIT)
        assert not found.fixable
        assert "would both become" in found.detail

    def test_two_spellings_of_one_value(self, dataset):
        """In two SUBJECTS, not two files in one folder: macOS and Windows
        are case-insensitive, so ``task-X`` and ``task-x`` side by side are
        one file there and the condition cannot be built that way."""
        (dataset / "sub-002/func").mkdir(parents=True)
        (dataset / "sub-002/func/sub-002_task-X_bold.nii.gz").write_bytes(b"x")
        found = [f for f in coherence.check(dataset)
                 if f.kind is Kind.ENTITY_CASE_SPLIT]
        assert found
        assert "task-X" in found[0].detail and "task-x" in found[0].detail
        assert not found[0].fixable, "which spelling is right is not ours"

    def test_a_task_name_that_does_not_derive_to_the_label(self, dataset):
        side = dataset / "sub-001/func/sub-001_task-x_bold.json"
        side.write_text(json.dumps({"TaskName": "something else"}))
        found = next(f for f in coherence.check(dataset)
                     if f.kind is Kind.TASK_NAME_MISMATCH)
        assert found.changes == (('TaskName: "something else"',
                                  'TaskName: "x"', 1),)
        coherence.apply(dataset, [found])
        assert json.loads(side.read_text())["TaskName"] == "x"

    def test_a_task_name_that_does_derive_is_left_alone(self, dataset):
        """The standard's own rule: the label is the name with everything
        but letters and digits removed."""
        side = dataset / "sub-001/func/sub-001_task-x_bold.json"
        side.write_text(json.dumps({"TaskName": "x"}))
        assert Kind.TASK_NAME_MISMATCH not in _kinds(coherence.check(dataset))

    def test_uneven_sessions(self, dataset):
        (dataset / "sub-002/ses-pre/anat").mkdir(parents=True)
        (dataset / "sub-002/ses-pre/anat/sub-002_ses-pre_T1w.nii.gz").write_bytes(b"x")
        found = [f for f in coherence.check(dataset)
                 if f.kind is Kind.SESSIONS_UNEVEN]
        assert found and not found[0].fixable

    def test_a_standalone_repair_runs_outside_the_batch(self, dataset):
        """A rename opens its own operation. Nesting one inside the batch
        would make the outer undo restore a half-renamed tree."""
        f = dataset / "sub-001/func/sub-001_task-x_bold.nii.gz"
        f.rename(f.with_name("sub-001_task-x_run-1_bold.nii.gz"))
        (dataset / "sub-001/func/sub-001_task-x_run-02_bold.nii.gz").write_bytes(b"x")
        found = [f for f in coherence.check(dataset)
                 if f.kind is Kind.ENTITY_WIDTH_SPLIT]
        assert coherence.apply(dataset, found) == 1
        assert (dataset / "sub-001/func/sub-001_task-x_run-01_bold.nii.gz").exists()


class TestARepairNeverWritesJunk:
    """The reported defect: it offered to add ``.DS_Store`` to scans.tsv.

    A repair WRITES into the dataset, so what it is about to write has to be
    a name the standard accepts. Two gates now: the shared recording
    predicate checks the suffix against the schema instead of assuming the
    shape of the name, and the coherence pass validates the whole basename
    before offering anything.
    """

    @pytest.mark.parametrize("junk", [
        ".DS_Store", "Thumbs.db", "desktop.ini", ".hidden", "notes.txt",
        "._resource", "README", "scratch.dat",
    ])
    def test_junk_is_never_a_recording(self, dataset, junk):
        (dataset / "sub-001/func" / junk).write_bytes(b"x")
        found = coherence.check(dataset)
        assert Kind.SCANS_ROW_MISSING not in _kinds(found), junk
        assert junk not in (dataset / "sub-001/sub-001_scans.tsv").read_text()

    def test_applying_with_junk_present_leaves_it_out(self, dataset):
        (dataset / "sub-001/func/.DS_Store").write_bytes(b"x")
        (dataset / "sub-001/func/sub-001_task-y_bold.nii.gz").write_bytes(b"x")
        coherence.apply(dataset, coherence.check(dataset))
        text = (dataset / "sub-001/sub-001_scans.tsv").read_text()
        assert "sub-001_task-y_bold.nii.gz" in text, "the real one was added"
        assert ".DS_Store" not in text

    def test_a_name_the_schema_rejects_is_not_offered(self, dataset):
        """Right extension, wrong name: an anatomical with no subject."""
        (dataset / "sub-001/anat/T1w.nii.gz").write_bytes(b"x")
        assert Kind.SCANS_ROW_MISSING not in _kinds(coherence.check(dataset))

    def test_a_real_recording_still_is_one(self, dataset):
        (dataset / "sub-001/func/sub-001_task-y_bold.nii.gz").write_bytes(b"x")
        assert Kind.SCANS_ROW_MISSING in _kinds(coherence.check(dataset))
