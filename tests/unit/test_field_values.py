"""Clearing a field must leave something the standard accepts.

Emptying a box used to write JSON ``null``. That is not "empty", it is a
value, and it is the one value BIDS never accepts. Measured against the
validator: ``{"Authors": null}`` is an ERROR, while ``{"Authors": []}`` and an
absent ``Authors`` are both clean. So tidying a file produced a violation.

What "empty" means comes from the field's declared type, which means it comes
from the schema and follows the schema version in force.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr import schema as sc
from bidsmgr.editor.field_values import (
    REMOVE,
    apply_empty,
    coerce_text,
    empty_value_for,
    is_empty_text,
)
from bidsmgr.editor.types import Severity
from bidsmgr.editor.validator import validate


def _spec(name: str, datatype: str = "", suffix: str = ""):
    fields = (
        sc.sidecar_fields(datatype, suffix) if datatype
        else sc.dataset_description_fields()
    )
    for field in fields:
        if field.name == name:
            return field
    raise AssertionError(f"{name} not declared")


# ---------------------------------------------------------------------------
# The empty form of each type
# ---------------------------------------------------------------------------


def test_an_array_empties_to_an_empty_list() -> None:
    """The reported case: clearing Authors wrote null."""
    assert empty_value_for(_spec("Authors")) == []


def test_a_string_empties_to_an_empty_string() -> None:
    assert empty_value_for(_spec("License")) == ""


def test_a_number_has_no_empty_form_so_the_key_goes() -> None:
    """There is no numeral meaning "unanswered", and null is an error."""
    assert empty_value_for(_spec("SamplingFrequency", "eeg", "eeg")) is REMOVE


def test_a_field_that_accepts_na_empties_to_na() -> None:
    """No empty form, but the standard does accept "no answer" here."""
    assert empty_value_for(
        _spec("PowerLineFrequency", "eeg", "eeg")
    ) == "n/a"


def test_a_field_nothing_describes_is_removed_not_invented() -> None:
    assert empty_value_for(None) is REMOVE


def test_a_multi_shape_field_empties_to_the_first_shape_listed() -> None:
    """IntendedFor is a string or an array of them; the schema lists the
    string first, so that is what clearing it means."""
    field = _spec("IntendedFor", "fmap", "phasediff")
    assert "string" in field.accepts and "array" in field.accepts
    assert empty_value_for(field) == ""


def test_the_empty_container_is_a_fresh_one_each_time() -> None:
    """A shared mutable default would let one file's edit reach another."""
    field = _spec("Authors")
    first, second = empty_value_for(field), empty_value_for(field)
    first.append("leak")
    assert second == []


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------


def test_clearing_writes_the_empty_form() -> None:
    data = {"Authors": ["A Person"]}
    assert apply_empty(data, "Authors", _spec("Authors"))
    assert data["Authors"] == []


def test_clearing_a_numeric_field_deletes_the_key() -> None:
    data = {"SamplingFrequency": 1000}
    assert apply_empty(
        data, "SamplingFrequency", _spec("SamplingFrequency", "eeg", "eeg"),
    )
    assert "SamplingFrequency" not in data


def test_clearing_something_already_clear_changes_nothing() -> None:
    data = {"Authors": []}
    assert not apply_empty(data, "Authors", _spec("Authors"))


def test_whitespace_counts_as_cleared() -> None:
    assert is_empty_text("") and is_empty_text("   ") and is_empty_text(None)
    assert not is_empty_text("0")


def test_the_word_null_is_not_a_null() -> None:
    """Somebody who types the word means the word. Turning it into a JSON
    null is how the thing this module prevents used to happen."""
    assert not is_empty_text("null")
    assert coerce_text("null", _spec("License")) == "null"


# ---------------------------------------------------------------------------
# Typing a value, with the schema deciding its shape
# ---------------------------------------------------------------------------


def test_a_numeric_field_takes_a_number_not_a_string() -> None:
    value = coerce_text("2048", _spec("SamplingFrequency", "eeg", "eeg"))
    assert value == 2048 and not isinstance(value, str)


def test_a_free_text_field_keeps_digits_as_text() -> None:
    """Guessing from the text is what turned "60" in a free-text field into
    the number 60."""
    assert coerce_text("60", _spec("License")) == "60"


def test_an_array_of_strings_splits_on_commas() -> None:
    assert coerce_text("A Person, Another", _spec("Authors")) == [
        "A Person", "Another",
    ]


def test_an_array_accepts_json_as_typed() -> None:
    assert coerce_text('["A", "B"]', _spec("Authors")) == ["A", "B"]


def test_clearing_through_coerce_uses_the_empty_form() -> None:
    assert coerce_text("   ", _spec("Authors")) == []
    assert coerce_text("", _spec("SamplingFrequency", "eeg", "eeg")) is REMOVE


# ---------------------------------------------------------------------------
# The property that made this worth doing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", ["Authors", "License"])
def test_clearing_a_field_does_not_create_a_validation_error(
    tmp_path: Path, field_name: str,
) -> None:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 16)
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text("{}")
    described = {
        "Name": "d", "BIDSVersion": "1.10.0",
        "Authors": ["A Person"], "License": "CC0",
    }
    path = root / "dataset_description.json"
    path.write_text(json.dumps(described))

    apply_empty(described, field_name, _spec(field_name))
    path.write_text(json.dumps(described))

    errors = [
        issue for verdict in validate(root).files
        for issue in (verdict.issues or [])
        if issue.severity is Severity.ERR
        and not getattr(issue, "mirrored", False)
    ]
    assert not errors, [(i.rule_id, i.field) for i in errors]


def test_null_is_what_we_are_avoiding(tmp_path: Path) -> None:
    """The regression, stated as a fact about the validator rather than as a
    belief about it. If this ever stops failing, the rule above is moot."""
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 16)
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text("{}")
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "d", "BIDSVersion": "1.10.0", "Authors": None,
    }))
    errors = [
        issue for verdict in validate(root).files
        for issue in (verdict.issues or [])
        if issue.severity is Severity.ERR
        and not getattr(issue, "mirrored", False)
    ]
    assert any(i.field == "Authors" for i in errors)
