"""What the dashboard counts, and the two things it must never do.

It answers the question a file tree cannot: what is in this dataset, is it
evenly filled, and where is the work. Two rules keep the numbers honest.

**Every figure is one a user could verify by hand.** No scores, no weighted
composites: "41 of 60 declared fields are answered" can be checked, and
"completeness 68%" can only be believed.

**An absence is reported as an absence.** A dataset with no participants table
gets a statement that there is no table, not a zero, because those are
different facts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import dashboard as dash
from bidsmgr.editor.validator import validate


def _make(root: Path, layout: dict[str, tuple[str, ...]]) -> Path:
    for sub, datatypes in layout.items():
        for datatype in datatypes:
            folder = root / sub / datatype
            folder.mkdir(parents=True, exist_ok=True)
            stem = {
                "anat": f"{sub}_T1w",
                "func": f"{sub}_task-rest_bold",
                "eeg": f"{sub}_task-rest_eeg",
            }[datatype]
            extension = ".edf" if datatype == "eeg" else ".nii.gz"
            (folder / f"{stem}{extension}").write_bytes(b"\0" * 1024)
            (folder / f"{stem}.json").write_text(json.dumps(
                {"TaskName": "rest"} if "task" in stem else {"EchoTime": 0.03}
            ))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Demo", "BIDSVersion": "1.10.0"})
    )
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = _make(tmp_path / "ds", {
        "sub-01": ("anat", "func", "eeg"),
        "sub-02": ("anat", "func"),
        "sub-03": ("anat", "func"),
        "sub-04": ("anat",),
    })
    (root / "participants.tsv").write_text(
        "participant_id\nsub-01\nsub-02\nsub-03\nsub-04\n"
    )
    return root


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_it_counts_what_is_there(dataset: Path) -> None:
    board = dash.build(dataset)
    assert board.name == "Demo"
    assert [s.label for s in board.subjects] == [
        "sub-01", "sub-02", "sub-03", "sub-04",
    ]
    assert {m.datatype for m in board.modalities} == {"anat", "func", "eeg"}
    assert board.total_files == 16
    assert board.total_bytes > 0


def test_a_modality_knows_how_many_subjects_have_it(dataset: Path) -> None:
    board = dash.build(dataset)
    by_name = {m.datatype: m for m in board.modalities}
    assert by_name["anat"].subjects == 4
    assert by_name["func"].subjects == 3
    assert by_name["eeg"].subjects == 1


def test_recordings_are_counted_apart_from_sidecars(dataset: Path) -> None:
    board = dash.build(dataset)
    by_name = {m.datatype: m for m in board.modalities}
    assert by_name["anat"].recordings == 4
    assert by_name["anat"].files == 8, "the sidecars are files too"


def test_coverage_is_answered_over_declared(dataset: Path) -> None:
    board = dash.build(dataset)
    for modality in board.modalities:
        assert modality.declared > 0
        assert 0 <= modality.answered <= modality.declared
        assert modality.coverage == modality.answered / modality.declared


def test_a_placeholder_counts_as_unanswered(tmp_path: Path) -> None:
    """Because that is what it is."""
    root = _make(tmp_path / "ds", {"sub-01": ("func",)})
    sidecar = root / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    # Both markers, on fields the schema actually declares for func/bold.
    # A field it does not declare is not counted at all, either way.
    sidecar.write_text(json.dumps({
        "TaskName": "rest",
        "InstitutionName": "TODO",
        "Manufacturer": "TODO",
        "NotAFieldAtAll": "TODO",
    }))
    board = dash.build(root)
    func = next(m for m in board.modalities if m.datatype == "func")
    assert func.placeholders == 2, "the two declared markers"
    assert func.answered == 1, "TaskName is a real answer"


# ---------------------------------------------------------------------------
# The insights, which are the point
# ---------------------------------------------------------------------------


def test_it_names_the_subject_missing_what_the_others_have(
    dataset: Path,
) -> None:
    """Invisible in a file tree, obvious in a table."""
    board = dash.build(dataset)
    assert board.uneven_subjects == ["sub-04"]


def test_one_over_complete_subject_does_not_indict_the_rest(
    dataset: Path,
) -> None:
    """sub-01 is the only one with EEG. That does not make the other three
    incomplete, so the comparison is against what a MAJORITY carry."""
    board = dash.build(dataset)
    assert "sub-02" not in board.uneven_subjects
    assert board.outlier_subjects == [("sub-01", ("eeg",))]


def test_an_even_dataset_flags_nobody(tmp_path: Path) -> None:
    root = _make(tmp_path / "ds", {
        "sub-01": ("anat", "func"), "sub-02": ("anat", "func"),
    })
    board = dash.build(root)
    assert board.uneven_subjects == []
    assert board.outlier_subjects == []


def test_one_subject_is_never_uneven(tmp_path: Path) -> None:
    """There is nothing to be uneven with."""
    board = dash.build(_make(tmp_path / "ds", {"sub-01": ("anat",)}))
    assert board.uneven_subjects == []


def test_the_repeated_rule_is_shown_as_one_mistake(dataset: Path) -> None:
    board = dash.build(dataset, validate(dataset))
    assert board.top_rules
    rule, count = board.top_rules[0]
    assert count > 1
    assert sum(n for _r, n in board.top_rules) <= board.errors + board.warnings


# ---------------------------------------------------------------------------
# Absences are absences
# ---------------------------------------------------------------------------


def test_no_participants_table_is_none_not_zero(tmp_path: Path) -> None:
    root = _make(tmp_path / "ds", {"sub-01": ("anat",)})
    assert dash.build(root).participants is None


def test_an_empty_participants_table_is_zero(tmp_path: Path) -> None:
    root = _make(tmp_path / "ds", {"sub-01": ("anat",)})
    (root / "participants.tsv").write_text("participant_id\n")
    assert dash.build(root).participants == 0


def test_a_readme_is_counted_once_however_it_is_spelled(
    tmp_path: Path,
) -> None:
    """Listing README, README.md and README.txt as three absences reads as
    three problems and is one."""
    root = _make(tmp_path / "ds", {"sub-01": ("anat",)})
    board = dash.build(root)
    assert board.absent.count("README") == 1
    assert "README.md" not in board.absent

    (root / "README.md").write_text("x" * 200)
    board = dash.build(root)
    assert "README" in board.present
    assert "README" not in board.absent


# ---------------------------------------------------------------------------
# It reads, and it does not validate
# ---------------------------------------------------------------------------


def test_it_never_writes(dataset: Path) -> None:
    before = {
        p: p.stat().st_mtime_ns for p in dataset.rglob("*") if p.is_file()
    }
    dash.build(dataset, validate(dataset))
    after = {
        p: p.stat().st_mtime_ns for p in dataset.rglob("*") if p.is_file()
    }
    assert before == after


def test_it_works_without_a_report(dataset: Path) -> None:
    """Drawing a summary is not a reason to revalidate a whole dataset."""
    board = dash.build(dataset)
    assert not board.validated
    assert board.errors == 0 and board.warnings == 0
    assert board.subjects, "the shape is still reported"


def test_counts_exclude_mirrors(dataset: Path) -> None:
    """A metadata finding shows on the sidecar and on the data file. Counting
    both would double every number here."""
    report = validate(dataset)
    board = dash.build(dataset, report)
    assert board.errors == report.counts["err"]
    assert board.warnings == report.counts["warn"]


def test_a_broken_dataset_does_not_crash_it(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text("{not json")
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text("{also not")
    board = dash.build(root)
    assert board.name == ""
    assert [s.label for s in board.subjects] == ["sub-01"]


def test_human_bytes_reads_as_a_size() -> None:
    assert dash.human_bytes(0) == "0 B"
    assert dash.human_bytes(999) == "999 B"
    assert dash.human_bytes(1536) == "1.5 KB"
    assert dash.human_bytes(5 * 1024 ** 3).endswith("GB")
