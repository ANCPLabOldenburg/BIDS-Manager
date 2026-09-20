"""Validation has to understand the inheritance principle, and it does.

A field stated once in a sidecar higher up applies to every file below it with
the same suffix and a compatible set of entities. Getting this wrong is
expensive in both directions: demanding a field that IS stated, higher up,
reports a violation that is not one; accepting a file because some unrelated
sidecar happens to mention the field accepts a dataset that is broken.

These tests pin the behaviour both ways, and pin what the Editor SHOWS for an
inherited value, because a form that displays effective metadata has to say
which file actually states it or editing makes a silent second copy.
"""

from __future__ import annotations

import json
from pathlib import Path


from bidsmgr.editor import inheritance as inh
from bidsmgr.editor.types import Severity
from bidsmgr.editor.validator import validate


def _dataset(root: Path, tasks: tuple[str, ...] = ("rest",)) -> Path:
    folder = root / "sub-01" / "func"
    folder.mkdir(parents=True)
    for task in tasks:
        (folder / f"sub-01_task-{task}_bold.nii.gz").write_bytes(b"\0" * 64)
        (folder / f"sub-01_task-{task}_bold.json").write_text(
            json.dumps({"EchoTime": 0.03})
        )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def _errors(root: Path) -> list[tuple[str, str, str]]:
    report = validate(root)
    return [
        (Path(v.path).name, i.rule_id, i.field or "")
        for v in report.files for i in (v.issues or [])
        if i.severity is Severity.ERR and not i.mirrored
    ]


def test_a_field_stated_higher_up_satisfies_the_requirement(
    tmp_path: Path,
) -> None:
    """The expensive mistake in one direction: reporting a violation for a
    field that IS stated, one level up."""
    root = _dataset(tmp_path / "ds")
    (root / "task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest", "RepetitionTime": 2.0})
    )
    assert _errors(root) == []


def test_the_same_dataset_without_the_inherited_file_reports_it(
    tmp_path: Path,
) -> None:
    """The control. If this did not fail, the test above would prove nothing."""
    root = _dataset(tmp_path / "ds")
    fields = {field for _name, _rule, field in _errors(root)}
    assert "TaskName" in fields
    assert "RepetitionTime" in fields


def test_inheritance_does_not_reach_a_file_it_does_not_describe(
    tmp_path: Path,
) -> None:
    """The expensive mistake in the other direction. A ``task-rest`` sidecar
    says nothing about a ``task-nback`` run, and accepting it would pass a
    dataset that is genuinely missing the field."""
    root = _dataset(tmp_path / "ds", tasks=("rest", "nback"))
    (root / "task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest", "RepetitionTime": 2.0})
    )
    errors = _errors(root)
    assert all("nback" in name for name, _rule, _field in errors), errors
    assert {field for _n, _r, field in errors} >= {
        "TaskName", "RepetitionTime",
    }


def test_the_inherited_file_is_itself_validated(tmp_path: Path) -> None:
    """It is a file in the dataset, not an invisible fragment."""
    root = _dataset(tmp_path / "ds")
    (root / "task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest", "RepetitionTime": 2.0})
    )
    seen = {Path(v.path).name for v in validate(root).files}
    assert "task-rest_bold.json" in seen


def test_the_nearest_sidecar_wins(tmp_path: Path) -> None:
    """And the dataset is valid either way: overriding is not an error."""
    root = _dataset(tmp_path / "ds")
    (root / "task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest", "RepetitionTime": 2.0})
    )
    child = root / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    child.write_text(json.dumps({"RepetitionTime": 3.0}))
    assert _errors(root) == []

    sources = inh.explain(root, child, "RepetitionTime")
    assert len(sources) == 2, "both files state it"
    winner = next(s for s in sources if s.winner)
    assert winner.value == 3.0
    assert winner.level == 0, "the nearest one"


def test_the_editor_can_say_where_an_inherited_value_comes_from(
    tmp_path: Path,
) -> None:
    """A form shows EFFECTIVE metadata. Without this, editing an inherited
    value silently makes a second copy of it."""
    root = _dataset(tmp_path / "ds")
    (root / "task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest", "RepetitionTime": 2.0})
    )
    child = root / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    sources = inh.explain(root, child, "TaskName")
    assert len(sources) == 1
    assert sources[0].rel == "task-rest_bold.json"
    assert sources[0].level > 0, "stated above, not here"
    assert sources[0].winner


def test_a_field_nobody_states_has_no_source(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    child = root / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    assert inh.explain(root, child, "InstitutionName") == []


def test_a_subject_level_sidecar_reaches_its_own_subject_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ds"
    for sub in ("sub-01", "sub-02"):
        folder = root / sub / "func"
        folder.mkdir(parents=True)
        (folder / f"{sub}_task-rest_bold.nii.gz").write_bytes(b"\0" * 64)
        (folder / f"{sub}_task-rest_bold.json").write_text(
            json.dumps({"EchoTime": 0.03})
        )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    (root / "sub-01" / "sub-01_task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest", "RepetitionTime": 2.0})
    )
    errors = _errors(root)
    assert all("sub-02" in name for name, _r, _f in errors), errors
