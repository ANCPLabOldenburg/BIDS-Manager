"""The Qt-free half of the Editor's automation.

Three engines, each tested against what it is for rather than against its
implementation:

* :mod:`bidsmgr.project.operations` - every write is reversible.
* :mod:`bidsmgr.editor.grouping` - a finding that fires 200 times is one row.
* :mod:`bidsmgr.editor.bulk_edit` - an answer reaches the files it belongs in,
  and no others.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bidsmgr.editor import bulk_edit as be
from bidsmgr.editor.grouping import group_report, summarise
from bidsmgr.editor.types import (
    FileVerdict,
    Issue,
    Severity,
    ValidationReport,
)
from bidsmgr.project.operations import (
    OperationError,
    begin_operation,
    read_log,
    undo_last,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A small two-subject dataset with MRI and EEG."""
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "func").mkdir(parents=True)
    (root / "sub-02" / "func").mkdir(parents=True)
    (root / "sub-01" / "eeg").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"EchoTime": 0.03})
    )
    (root / "sub-01" / "func" / "sub-01_task-rest_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"})
    )
    (root / "sub-02" / "func" / "sub-02_task-rest_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"})
    )
    (root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.json").write_text(
        json.dumps({"SamplingFrequency": 500})
    )
    return root


def _issue(rule: str, field: str, sev: Severity = Severity.WARN) -> Issue:
    return Issue(
        severity=sev, rule_id=rule, message=f"{field} is a problem",
        field=field,
    )


# ---------------------------------------------------------------------------
# operations: nothing is written that cannot be taken back
# ---------------------------------------------------------------------------


def test_a_write_can_be_undone(dataset: Path) -> None:
    target = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    with begin_operation(dataset, "Set Manufacturer") as op:
        op.write_json(target, {"EchoTime": 0.03, "Manufacturer": "Siemens"})
    assert json.loads(target.read_text())["Manufacturer"] == "Siemens"
    assert len(read_log(dataset)) == 1

    undo_last(dataset)
    assert "Manufacturer" not in json.loads(target.read_text())
    assert read_log(dataset) == []


def test_undo_removes_a_file_the_operation_created(dataset: Path) -> None:
    """Undo of a create is a delete, not a restore of nothing."""
    created = dataset / "sub-01" / "func" / "sub-01_task-rest_events.tsv"
    with begin_operation(dataset, "Generate events") as op:
        op.write_text(created, "onset\tduration\n")
    assert created.exists()
    undo_last(dataset)
    assert not created.exists()


def test_a_failure_part_way_through_changes_nothing(dataset: Path) -> None:
    """Two writes, the second raises: the first must not survive."""
    a = dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    b = dataset / "sub-02" / "func" / "sub-02_task-rest_bold.json"
    before_a = a.read_text()
    before_b = b.read_text()
    with pytest.raises(ValueError):
        with begin_operation(dataset, "Two writes") as op:
            op.write_json(a, {"RepetitionTime": 99})
            op.write_json(b, {"RepetitionTime": 99})
            raise ValueError("something went wrong")
    assert a.read_text() == before_a
    assert b.read_text() == before_b
    assert read_log(dataset) == []


def test_an_operation_that_changed_nothing_leaves_no_history(
    dataset: Path,
) -> None:
    with begin_operation(dataset, "Did nothing"):
        pass
    assert read_log(dataset) == []


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "the premise cannot be set up on Windows: os.chmod there only toggles "
        "the read-only attribute, and Windows ignores that attribute for "
        "directories, so the root stays writable and there is nothing to "
        "refuse. Making a directory unwritable needs an ACL change, which is "
        "more test scaffolding than the assertion is worth."
    ),
)
def test_a_read_only_root_is_refused_rather_than_half_written(
    dataset: Path,
) -> None:
    """The history lives in the dataset, so an unwritable one cannot be
    edited safely. Say so instead of failing on the first write."""
    import stat

    mode = dataset.stat().st_mode
    os.chmod(dataset, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(OperationError):
            begin_operation(dataset, "Nope")
    finally:
        os.chmod(dataset, mode)


# ---------------------------------------------------------------------------
# grouping: one row per kind of finding
# ---------------------------------------------------------------------------


def test_the_same_finding_across_files_is_one_group() -> None:
    report = ValidationReport(
        files=[
            FileVerdict(path=Path(f"sub-{i:02d}/func/x_bold.json"),
                        issues=[_issue("TSV_COLUMN_MISSING", "units")])
            for i in range(1, 6)
        ],
    )
    groups = group_report(report)
    assert len(groups) == 1
    assert groups[0].count == 5
    assert len(groups[0].files) == 5
    assert "5 findings in 1 kinds" in summarise(groups)


def test_the_same_rule_on_a_different_field_is_a_different_group() -> None:
    """The fix differs, so the row must differ."""
    report = ValidationReport(
        files=[
            FileVerdict(path=Path("a.json"),
                        issues=[_issue("SIDECAR_KEY_RECOMMENDED", "EchoTime")]),
            FileVerdict(path=Path("b.json"),
                        issues=[_issue("SIDECAR_KEY_RECOMMENDED", "FlipAngle")]),
        ],
    )
    assert len(group_report(report)) == 2


def test_mirrored_findings_are_not_counted_twice() -> None:
    """A data file's finding is mirrored onto its sidecar for display. Counting
    both would double every metadata finding."""
    real = _issue("SIDECAR_KEY_RECOMMENDED", "EchoTime")
    mirror = _issue("SIDECAR_KEY_RECOMMENDED", "EchoTime")
    mirror.mirrored = True
    report = ValidationReport(
        files=[
            FileVerdict(path=Path("x_bold.nii.gz"), issues=[real]),
            FileVerdict(path=Path("x_bold.json"), issues=[mirror]),
        ],
    )
    groups = group_report(report)
    assert len(groups) == 1
    assert groups[0].count == 1


def test_errors_sort_above_warnings() -> None:
    report = ValidationReport(
        files=[
            FileVerdict(path=Path("a.json"),
                        issues=[_issue("W", "f1", Severity.WARN)]),
            FileVerdict(path=Path("b.json"),
                        issues=[_issue("E", "f2", Severity.ERR)]),
        ],
    )
    groups = group_report(report)
    assert groups[0].rule_id == "E"


# ---------------------------------------------------------------------------
# bulk_edit: the right files, and only the right files
# ---------------------------------------------------------------------------


def test_scope_same_kind_finds_the_matching_suffix(dataset: Path) -> None:
    anchor = dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    cands = be.candidates(
        dataset, "RepetitionTime", anchor=anchor, scope=be.SCOPE_SAME_KIND,
    )
    rels = {c.rel for c in cands}
    assert "sub-02/func/sub-02_task-rest_bold.json" in rels
    assert "sub-01/anat/sub-01_T1w.json" not in rels


def test_a_field_the_datatype_does_not_declare_is_offered_but_refused(
    dataset: Path,
) -> None:
    """Shown with a reason rather than hidden: a user who expected the file
    there deserves to know why it will not be written."""
    cands = be.candidates(dataset, "EchoTime", scope=be.SCOPE_DATASET)
    eeg = [c for c in cands if c.rel.endswith("_eeg.json")]
    assert eeg and not eeg[0].applicable
    assert "does not declare" in eeg[0].reason


def test_dataset_level_files_are_never_candidates(dataset: Path) -> None:
    """dataset_description.json has its own schema. A bulk edit reaching it
    would write an acquisition field into the dataset description."""
    cands = be.candidates(dataset, "Manufacturer", scope=be.SCOPE_DATASET)
    dd = [c for c in cands if c.rel == "dataset_description.json"]
    assert dd and not dd[0].applicable
    assert "dataset-level" in dd[0].reason


def test_apply_writes_only_the_selection_and_is_one_undo(
    dataset: Path,
) -> None:
    cands = [
        c for c in be.candidates(
            dataset, "InstitutionName", scope=be.SCOPE_DATASET,
        )
        if c.applicable and c.rel.endswith("_bold.json")
    ]
    assert len(cands) == 2
    result = be.apply_value(dataset, cands, "InstitutionName", "Oldenburg")
    assert len(result.written) == 2
    assert result.ok

    for c in cands:
        assert json.loads(c.path.read_text())["InstitutionName"] == "Oldenburg"
    # The T1w was applicable but not selected.
    t1w = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    assert "InstitutionName" not in json.loads(t1w.read_text())

    # One operation, so one undo puts both back.
    assert len(read_log(dataset)) == 1
    undo_last(dataset)
    for c in cands:
        assert "InstitutionName" not in json.loads(c.path.read_text())


def test_a_file_that_already_says_it_is_not_rewritten(dataset: Path) -> None:
    """A count of "12 files" should mean twelve files that will differ."""
    target = dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    cands = be.candidates(dataset, "TaskName", paths=[target])
    assert cands[0].current == "rest"
    assert not cands[0].would_change("rest")
    result = be.apply_value(dataset, cands, "TaskName", "rest")
    assert result.written == []
    assert result.skipped == [target]


def test_a_finding_on_a_data_file_edits_its_sidecar(dataset: Path) -> None:
    """Findings attach to the data file; the field is edited in the sidecar."""
    nii = dataset / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz"
    nii.write_bytes(b"")
    cands = be.candidates(dataset, "RepetitionTime", paths=[nii])
    assert len(cands) == 1
    assert cands[0].path.name == "sub-01_task-rest_bold.json"
    assert cands[0].current == 2.0


# ---------------------------------------------------------------------------
# Repairing a value's shape: the rule is that a repair may change a TYPE,
# never invent a FACT.
# ---------------------------------------------------------------------------


def test_a_number_written_as_text_is_repaired(dataset: Path) -> None:
    target = dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    target.write_text(json.dumps({"RepetitionTime": "2.0", "TaskName": "rest"}))
    plan = be.plan_coercions(dataset, "RepetitionTime")
    assert [c.path for c in plan] == [target]
    assert "number" in plan[0].reason
    be.apply_coercions(dataset, plan, "RepetitionTime")
    assert json.loads(target.read_text())["RepetitionTime"] == 2.0


def test_an_array_of_numbers_written_as_strings_is_repaired(
    dataset: Path,
) -> None:
    """The defect that made the metadata dialog add errors, as a fix action."""
    target = dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    target.write_text(json.dumps({"SliceTiming": ["0", "0.5"]}))
    plan = be.plan_coercions(dataset, "SliceTiming")
    be.apply_coercions(dataset, plan, "SliceTiming")
    assert json.loads(target.read_text())["SliceTiming"] == [0.0, 0.5]


def test_a_value_that_cannot_be_converted_is_left_alone(
    dataset: Path,
) -> None:
    """Losing a value to make a validator quiet is the worst outcome here."""
    target = dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    target.write_text(json.dumps({"RepetitionTime": "about two"}))
    assert be.plan_coercions(dataset, "RepetitionTime") == []
    assert json.loads(target.read_text())["RepetitionTime"] == "about two"


def test_a_value_already_the_right_type_is_not_offered(dataset: Path) -> None:
    plan = be.plan_coercions(dataset, "RepetitionTime")
    assert plan == []


# ---------------------------------------------------------------------------
# Tabular repairs
# ---------------------------------------------------------------------------


@pytest.fixture
def events(dataset: Path) -> Path:
    p = dataset / "sub-01" / "func" / "sub-01_task-rest_events.tsv"
    p.write_text("trial_type\tonset\tduration\ngo\t1.5\t0.2\nstop\tbad\t0.2\n")
    return p


def test_a_missing_column_can_be_added(dataset: Path, events: Path) -> None:
    from bidsmgr.editor import tsv_edit as te

    plan = te.plan_add_column(dataset, [events], "response_time")
    assert plan[0].applicable and plan[0].n_cells == 2
    te.apply_add_column(dataset, plan, "response_time")
    table = te.read_table(events)
    assert "response_time" in table.header
    assert table.column("response_time") == ["n/a", "n/a"]


def test_a_column_that_is_already_there_is_refused(
    dataset: Path, events: Path,
) -> None:
    from bidsmgr.editor import tsv_edit as te

    plan = te.plan_add_column(dataset, [events], "onset")
    assert not plan[0].applicable
    assert "already has" in plan[0].reason


def test_only_the_cells_that_fail_their_type_are_touched(
    dataset: Path, events: Path,
) -> None:
    from bidsmgr.editor import tsv_edit as te

    events.write_text("onset\tduration\n1.50\t0.2\nbad\t0.2\n")
    plan = te.plan_coerce_column(dataset, [events], "onset")
    assert plan[0].applicable
    assert "cannot be converted" in plan[0].detail
    te.apply_coerce_column(dataset, plan, "onset")
    # 1.50 normalised, "bad" left exactly as it was.
    assert te.read_table(events).column("onset") == ["1.5", "bad"]


def test_reordering_moves_no_data(dataset: Path, events: Path) -> None:
    from bidsmgr.editor import tsv_edit as te

    before = {
        row[0]: row for row in te.read_table(events).rows
    }
    plan = te.plan_reorder(dataset, [events])
    assert plan[0].applicable
    te.apply_reorder(dataset, plan)
    table = te.read_table(events)
    assert table.header[:3] == ["onset", "duration", "trial_type"]
    # Every row's values travelled with their column.
    for row in table.rows:
        original = before[row[2]]
        assert row == [original[1], original[2], original[0]]


def test_a_table_already_in_order_is_not_rewritten(
    dataset: Path, events: Path,
) -> None:
    from bidsmgr.editor import tsv_edit as te

    events.write_text("onset\tduration\ttrial_type\n1.0\t0.2\tgo\n")
    plan = te.plan_reorder(dataset, [events])
    assert not plan[0].applicable
    assert plan[0].reason == "already in order"


def test_a_tabular_repair_is_one_undo(dataset: Path, events: Path) -> None:
    from bidsmgr.editor import tsv_edit as te

    before = events.read_text()
    plan = te.plan_add_column(dataset, [events], "response_time")
    te.apply_add_column(dataset, plan, "response_time")
    assert len(read_log(dataset)) == 1
    undo_last(dataset)
    assert events.read_text() == before
