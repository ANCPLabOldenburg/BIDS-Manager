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
    NO_TODO,
    todo_value_for,
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
    assert todo_value_for(_spec("eeg", "eeg", "TaskName")) == "TODO"


def test_a_field_that_accepts_na_takes_na() -> None:
    """The schema says so explicitly, and it is how BIDS itself spells
    "this was asked and there is no answer". The old rule saw ``anyOf`` with
    no plain type and skipped the field entirely."""
    field = _spec("eeg", "eeg", "PowerLineFrequency")
    assert field.accepts == ("number", "string")
    assert field.accepts_na
    assert not field.accepts_free_text, "its string variant is only 'n/a'"
    assert todo_value_for(field) == "n/a"


@pytest.mark.parametrize("datatype, suffix, name", [
    ("fmap", "epi", "B0FieldIdentifier"), ("func", "bold", "B0FieldSource"),
    ("func", "sbref", "B0FieldSource"),
])
def test_a_field_that_links_files_gets_no_marker(datatype, suffix, name) -> None:
    """The same "TODO" on a fieldmap and a run IS a link for sdcflows and
    fMRIPrep (measured 2026-10-12 on the Shayam and MRI_sample_2
    conversions): a run with no fieldmap must say nothing, not name one."""
    field = _spec(datatype, suffix, name)
    assert todo_value_for(field) is NO_TODO
    assert "link" in unmarkable_reason(field)


def test_an_array_of_strings_takes_a_marker_in_a_list() -> None:
    field = next(
        f for f in sc.dataset_description_fields() if f.name == "Authors"
    )
    assert todo_value_for(field) == ["TODO"]


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
    assert todo_value_for(field) is NO_TODO


def test_a_real_vocabulary_gets_no_marker() -> None:
    """Nothing in a controlled vocabulary means "unanswered"."""
    field = _spec("anat", "T1w", "MRAcquisitionType")
    assert field.enum
    assert todo_value_for(field) is NO_TODO


def test_every_unmarkable_field_says_why() -> None:
    for datatype, suffix in (("anat", "T1w"), ("pet", "pet"), ("eeg", "eeg")):
        for field in sc.sidecar_fields(datatype, suffix):
            reason = unmarkable_reason(field)
            if todo_value_for(field) is NO_TODO:
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
        dataset, fill_scope=scope, write_report=False,
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
            copy, fill_scope=scope, write_report=False,
        )
        counts[scope] = sum(len(f.fields) for f in report.todo_fills)
    assert counts[FILL_REQUIRED] < counts[FILL_RECOMMENDED]
    assert counts[FILL_RECOMMENDED] < counts[FILL_OPTIONAL]


def test_the_fill_todos_flag_means_the_recommended_scope(
    dataset: Path, tmp_path: Path,
) -> None:
    """``bidsmgr-metadata --fill-todos`` keeps doing what it always did."""
    import shutil

    from bidsmgr.cli.metadata import main

    copy = tmp_path / "copy"
    shutil.copytree(dataset, copy)
    assert main([str(dataset), "--fill-todos", "--no-log-file"]) == 0
    assert main([str(copy), "--fill-scope", "recommended", "--no-log-file"]) == 0
    a = json.loads((dataset / ".bidsmgr" / "metadata_report.json").read_text())
    b = json.loads((copy / ".bidsmgr" / "metadata_report.json").read_text())
    assert a["fill_scope"] == FILL_RECOMMENDED
    assert sum(len(f["fields"]) for f in a["todo_fills"]) == \
        sum(len(f["fields"]) for f in b["todo_fills"])


def test_filling_nothing_writes_nothing(dataset: Path) -> None:
    before = {
        p: p.read_bytes() for p in dataset.rglob("*.json") if p.is_file()
    }
    run_metadata(dataset, write_report=False)
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
        dataset, fill_scope=FILL_OPTIONAL, write_report=False,
    )
    after = json.loads(target.read_text())
    assert after["TaskName"] == "resting state"
    assert after["PowerLineFrequency"] == 50
    assert after["EEGReference"] == "Cz"


def test_what_could_not_be_marked_is_reported(dataset: Path) -> None:
    report = run_metadata(
        dataset, fill_scope=FILL_RECOMMENDED,
        write_report=False,
    )
    assert report.unmarkable, "a partial fill must say what it left"
    assert any("EchoTime" in line for line in report.unmarkable)
    assert all("(" in line for line in report.unmarkable), "each gives a reason"


def test_an_invalid_scope_is_refused(dataset: Path) -> None:
    with pytest.raises(ValueError, match="fill_scope"):
        run_metadata(dataset, fill_scope="everything")


# ---------------------------------------------------------------------------
# The placeholders follow the scope, both ways
# ---------------------------------------------------------------------------


def _todos_by_level(root: Path) -> dict[str, set[str]]:
    """Every TODO placeholder in the dataset, grouped by the level the Editor's
    form shows its field at: the user's view of "TODOs in optional"."""
    from bidsmgr.editor.bidsmgr_checks import sidecar_fields_for
    from bidsmgr.metadata.engine import _infer_datatype_suffix

    out: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.json")):
        if ".bidsmgr" in path.parts:
            continue
        datatype, suffix = _infer_datatype_suffix(path, root)
        for field in sidecar_fields_for(path, datatype, suffix, root):
            if field.value in ("TODO", ["TODO"]):
                out.setdefault(field.level.value, set()).add(
                    f"{path.relative_to(root)}:{field.name}"
                )
    return out


@pytest.mark.parametrize("scope, allowed", [
    (FILL_REQUIRED, {"required"}),
    (FILL_RECOMMENDED, {"required", "recommended"}),
    (FILL_OPTIONAL, {"required", "recommended", "optional"}),
])
def test_every_placeholder_sits_at_a_level_the_scope_covers(
    dataset: Path, scope: str, allowed: set[str],
) -> None:
    """What the user checks in the Editor: choose "required" and no TODO may
    sit in a recommended or optional field. Run after a wider run, because a
    dataset is converted and marked more than once."""
    run_metadata(dataset, fill_scope=FILL_OPTIONAL, write_report=False)
    run_metadata(dataset, fill_scope=scope, write_report=False)
    found = _todos_by_level(dataset)
    assert set(found) <= allowed, {
        level: sorted(names)[:5] for level, names in found.items()
        if level not in allowed
    }


def test_narrowing_the_scope_takes_back_what_a_wider_one_wrote(
    dataset: Path,
) -> None:
    wide = run_metadata(dataset, fill_scope=FILL_OPTIONAL, write_report=False)
    written = {(f.sidecar, name) for f in wide.todo_fills for name in f.fields}
    narrow = run_metadata(dataset, fill_scope=FILL_RECOMMENDED, write_report=False)
    taken = {(f.sidecar, name) for f in narrow.todo_removed for name in f.fields}
    assert taken, "the optional placeholders must come back out"
    assert taken <= written
    assert narrow.fill_scope == FILL_RECOMMENDED
    # Running the same scope again changes nothing.
    again = run_metadata(dataset, fill_scope=FILL_RECOMMENDED, write_report=False)
    assert not again.todo_removed and not again.todo_fills


def test_none_takes_back_every_todo_and_leaves_values_and_na(dataset: Path) -> None:
    target = dataset / "sub-01" / "eeg" / "sub-01_task-rest_eeg.json"
    run_metadata(dataset, fill_scope=FILL_OPTIONAL, write_report=False)
    data = json.loads(target.read_text())
    data["EEGReference"] = "Cz"                 # a value the user typed
    data["MyLabNote"] = "TODO"                  # a key the standard does not declare
    target.write_text(json.dumps(data))
    na_before = {k for k, v in data.items() if v == "n/a"}

    run_metadata(dataset, fill_scope="none", write_report=False)
    assert _todos_by_level(dataset) == {
        "optional": {"sub-01/eeg/sub-01_task-rest_eeg.json:MyLabNote"},
    }
    after = json.loads(target.read_text())
    assert after["EEGReference"] == "Cz"
    assert after["MyLabNote"] == "TODO", "not ours to take back"
    # n/a is how a converter answers, so it is never taken back.
    assert na_before <= {k for k, v in after.items() if v == "n/a"}


def test_no_scope_leaves_placeholders_alone(dataset: Path) -> None:
    run_metadata(dataset, fill_scope=FILL_OPTIONAL, write_report=False)
    before = _todos_by_level(dataset)
    report = run_metadata(dataset, write_report=False)
    assert _todos_by_level(dataset) == before
    assert report.fill_scope == "unchanged"
    assert not report.todo_removed and not report.todo_fills


def test_a_field_that_names_a_file_takes_no_marker(tmp_path: Path) -> None:
    """"TODO" in AssociatedEmptyRoom names a recording that does not exist,
    which the validator reports as an error on every MEG file. A marker an
    earlier run wrote there is taken back."""
    from bidsmgr.editor.rename import PATH_FIELDS

    root = tmp_path / "ds"
    folder = root / "sub-01" / "meg"
    folder.mkdir(parents=True)
    (folder / "sub-01_task-rest_meg.fif").write_bytes(b"\0" * 16)
    sidecar = folder / "sub-01_task-rest_meg.json"
    sidecar.write_text(json.dumps({"AssociatedEmptyRoom": "TODO"}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    report = run_metadata(root, fill_scope=FILL_OPTIONAL, write_report=False)
    data = json.loads(sidecar.read_text())
    assert not set(PATH_FIELDS) & set(data), data
    assert any("AssociatedEmptyRoom" in line and "names another file" in line
               for line in report.unmarkable)
    for name in PATH_FIELDS:
        assert todo_value_for(type("F", (), {"name": name, "type": "string"})()) \
            is NO_TODO, name
