"""What the placeholder fill marks, and the one thing it may never do.

A placeholder marks a gap so it is visible in the file and reported by
validation instead of being an absence nobody notices. That only works if the
marker is itself valid: writing ``"TODO"`` into a numeric field turns a field
that was merely missing into a type ERROR, which is worse than the gap.

So there are two properties, and they pull against each other:

* cover as much of what is missing as possible;
* never introduce a validation error.

The second wins whenever they conflict, and the fields that cannot be marked
are reported rather than skipped silently, so a fill that covers two thirds
cannot report itself as complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr import schema as sc
from bidsmgr.editor.types import Severity
from bidsmgr.editor.validator import validate
from bidsmgr.metadata.engine import (
    FILL_OPTIONAL,
    FILL_RECOMMENDED,
    FILL_REQUIRED,
    FILL_SCOPES,
    _NO_TODO,
    _todo_value_for,
    run_metadata,
    scope_levels,
    unmarkable_reason,
)


def _spec(datatype: str, suffix: str, name: str):
    for field in sc.sidecar_fields(datatype, suffix):
        if field.name == name:
            return field
    raise AssertionError(f"{name} is not declared for {datatype}/{suffix}")


# ---------------------------------------------------------------------------
# Which marker a field gets, and why
# ---------------------------------------------------------------------------


def test_a_free_text_field_takes_the_todo_marker() -> None:
    assert _todo_value_for(_spec("eeg", "eeg", "TaskName")) == "TODO"


def test_a_field_that_accepts_na_takes_na() -> None:
    """The schema says so explicitly, and it is how BIDS itself spells
    "this was asked and there is no answer". The old rule saw ``anyOf`` with
    no plain type and skipped the field entirely."""
    field = _spec("eeg", "eeg", "PowerLineFrequency")
    assert field.accepts == ("number", "string")
    assert field.accepts_na
    assert not field.accepts_free_text, "its string variant is only 'n/a'"
    assert _todo_value_for(field) == "n/a"


def test_an_array_of_strings_takes_a_marker_in_a_list() -> None:
    field = next(
        f for f in sc.dataset_description_fields() if f.name == "Authors"
    )
    assert _todo_value_for(field) == ["TODO"]


@pytest.mark.parametrize("name", [
    "EchoTime", "FlipAngle", "NumberShots", "RepetitionTimePreparation",
])
def test_a_number_or_array_of_numbers_gets_no_marker(name: str) -> None:
    """The regression this rule exists for. These are ``anyOf`` number or
    array-of-number, and the array branch carries its own ``items``. Reading
    only the top level reported no item type, and treating that as "probably
    strings" wrote ``["TODO"]`` into them, turning a gap into a type error."""
    field = _spec("anat", "T1w", name)
    assert field.item_type == "number", "the item type must survive anyOf"
    assert _todo_value_for(field) is _NO_TODO


def test_a_real_vocabulary_gets_no_marker() -> None:
    """Nothing in a controlled vocabulary means "unanswered"."""
    field = _spec("anat", "T1w", "MRAcquisitionType")
    assert field.enum
    assert _todo_value_for(field) is _NO_TODO


def test_every_unmarkable_field_says_why() -> None:
    for datatype, suffix in (("anat", "T1w"), ("pet", "pet"), ("eeg", "eeg")):
        for field in sc.sidecar_fields(datatype, suffix):
            reason = unmarkable_reason(field)
            if _todo_value_for(field) is _NO_TODO:
                assert reason, f"{field.name} is skipped with no reason"
            else:
                assert not reason


def test_a_markable_field_reports_no_reason() -> None:
    assert unmarkable_reason(_spec("eeg", "eeg", "TaskName")) == ""


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


def test_the_scopes_nest() -> None:
    assert scope_levels(FILL_REQUIRED) == {"required"}
    assert scope_levels(FILL_RECOMMENDED) == {"required", "recommended"}
    assert scope_levels(FILL_OPTIONAL) == {
        "required", "recommended", "optional",
    }
    assert scope_levels("none") == frozenset()


def test_deprecated_is_in_no_scope() -> None:
    """A placeholder in a field BIDS is retiring is work nobody should do."""
    for scope in FILL_SCOPES:
        assert "deprecated" not in scope_levels(scope)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for datatype, stem, ext in (
        ("anat", "sub-01_T1w", ".nii.gz"),
        ("func", "sub-01_task-rest_bold", ".nii.gz"),
        ("eeg", "sub-01_task-rest_eeg", ".edf"),
    ):
        folder = root / "sub-01" / datatype
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{stem}{ext}").write_bytes(b"\0" * 16)
        (folder / f"{stem}.json").write_text(
            json.dumps({"TaskName": "rest"} if "task" in stem else {})
        )
    (root / "sub-01" / "func" / "sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\n1.0\t0.5\n"
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def _errors(root: Path) -> list:
    report = validate(root)
    return [
        issue for verdict in report.files for issue in (verdict.issues or [])
        if issue.severity is Severity.ERR
        and not getattr(issue, "mirrored", False)
    ]


@pytest.mark.parametrize("scope", [
    FILL_REQUIRED, FILL_RECOMMENDED, FILL_OPTIONAL,
])
def test_filling_never_adds_a_validation_error(
    dataset: Path, scope: str,
) -> None:
    """The property that matters most. Filling gaps must not manufacture
    violations, at any scope."""
    before = len(_errors(dataset))
    run_metadata(
        dataset, fill_todos=True, fill_scope=scope, write_report=False,
    )
    after = _errors(dataset)
    assert len(after) <= before, [
        (i.rule_id, i.field) for i in after
    ]
    assert not [i for i in after if i.rule_id == "JSON_SCHEMA_VALIDATION_ERROR"]


def test_a_wider_scope_marks_strictly_more(dataset: Path, tmp_path: Path) -> None:
    import shutil

    counts = {}
    for scope in (FILL_REQUIRED, FILL_RECOMMENDED, FILL_OPTIONAL):
        copy = tmp_path / f"copy_{scope}"
        shutil.copytree(dataset, copy)
        report = run_metadata(
            copy, fill_todos=True, fill_scope=scope, write_report=False,
        )
        counts[scope] = sum(len(f.fields) for f in report.todo_fills)
    assert counts[FILL_REQUIRED] < counts[FILL_RECOMMENDED]
    assert counts[FILL_RECOMMENDED] < counts[FILL_OPTIONAL]


def test_the_default_scope_is_what_the_flag_always_meant(
    dataset: Path, tmp_path: Path,
) -> None:
    """``fill_todos=True`` with no scope must keep doing what it did."""
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(dataset, copy)
    a = run_metadata(dataset, fill_todos=True, write_report=False)
    b = run_metadata(
        copy, fill_todos=True, fill_scope=FILL_RECOMMENDED, write_report=False,
    )
    assert a.fill_scope == FILL_RECOMMENDED
    assert sum(len(f.fields) for f in a.todo_fills) == \
        sum(len(f.fields) for f in b.todo_fills)


def test_filling_nothing_writes_nothing(dataset: Path) -> None:
    before = {
        p: p.read_bytes() for p in dataset.rglob("*.json") if p.is_file()
    }
    run_metadata(dataset, fill_todos=False, write_report=False)
    for path, text in before.items():
        if path.name == "dataset_description.json":
            continue   # the engine legitimately stamps GeneratedBy there
        assert path.read_bytes() == text, path


def test_an_existing_value_is_never_replaced(dataset: Path) -> None:
    """At any scope. This is the one guarantee a user has to be able to rely
    on before letting a fill near curated metadata."""
    target = dataset / "sub-01" / "eeg" / "sub-01_task-rest_eeg.json"
    target.write_text(json.dumps({
        "TaskName": "resting state",
        "PowerLineFrequency": 50,
        "EEGReference": "Cz",
    }))
    run_metadata(
        dataset, fill_todos=True, fill_scope=FILL_OPTIONAL, write_report=False,
    )
    after = json.loads(target.read_text())
    assert after["TaskName"] == "resting state"
    assert after["PowerLineFrequency"] == 50
    assert after["EEGReference"] == "Cz"


def test_what_could_not_be_marked_is_reported(dataset: Path) -> None:
    report = run_metadata(
        dataset, fill_todos=True, fill_scope=FILL_RECOMMENDED,
        write_report=False,
    )
    assert report.unmarkable, "a partial fill must say what it left"
    assert any("EchoTime" in line for line in report.unmarkable)
    assert all("(" in line for line in report.unmarkable), "each gives a reason"


def test_an_invalid_scope_is_refused(dataset: Path) -> None:
    with pytest.raises(ValueError, match="fill_scope"):
        run_metadata(dataset, fill_todos=True, fill_scope="everything")
