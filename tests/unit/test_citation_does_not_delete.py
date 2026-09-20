"""Writing a citation file must not silently delete what somebody typed.

The report: "filling author in template does not write into the json file...
maybe because CITATION.cff is created so author go there?" That was right, and
it was worse than reported: the writer moved FOUR fields out of
dataset_description.json when the standard forbids duplicating one.

Measured against the validator, one field at a time, with a CITATION.cff
present:

    Authors             -> AUTHORS_AND_CITATION_FILE_MUTUALLY_EXCLUSIVE
    License             -> no finding
    HowToAcknowledge    -> no finding
    ReferencesAndLinks  -> no finding

So a user who typed MIT into the licence had it written correctly by the
conversion and then deleted by the citation step. One field moves; the rest
are copied into the citation file and left where they are.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor.cff import COPIED_FIELDS, MOVED_FIELDS
from bidsmgr.editor.types import Severity
from bidsmgr.editor.validator import validate
from bidsmgr.fixups.citation import fields_moved_by_citation, write_citation

FILLED = {
    "Name": "Sleep and memory",
    "BIDSVersion": "1.10.0",
    "Authors": ["Karel Lopez"],
    "License": "MIT",
    "HowToAcknowledge": "Please cite the paper.",
    "ReferencesAndLinks": ["https://example.org/paper"],
}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    folder = root / "sub-01" / "anat"
    folder.mkdir(parents=True)
    (folder / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 64)
    (folder / "sub-01_T1w.json").write_text("{}")
    (root / "dataset_description.json").write_text(json.dumps(FILLED))
    return root


def _described(root: Path) -> dict:
    return json.loads((root / "dataset_description.json").read_text())


def test_only_authors_is_taken_out(dataset: Path) -> None:
    write_citation(dataset)
    after = _described(dataset)
    assert "Authors" not in after, "the one field the standard forbids in both"
    for name in COPIED_FIELDS:
        assert name in after, f"{name} was removed and should not have been"


def test_the_licence_a_user_typed_survives(dataset: Path) -> None:
    """The reported case, in one line."""
    write_citation(dataset)
    assert _described(dataset)["License"] == "MIT"


def test_everything_still_reaches_the_citation_file(dataset: Path) -> None:
    """Copied, not moved: the citation file is a rendering of these, not
    their new home."""
    write_citation(dataset)
    text = (dataset / "CITATION.cff").read_text()
    assert "MIT" in text
    assert "Lopez" in text


def test_the_result_validates_clean(dataset: Path) -> None:
    write_citation(dataset)
    errors = [
        i for v in validate(dataset).files for i in (v.issues or [])
        if i.severity is Severity.ERR and not i.mirrored
    ]
    assert not errors, [(i.rule_id, i.field) for i in errors]


def test_keeping_authors_would_be_an_error(dataset: Path) -> None:
    """Stated as a fact about the validator, so if it ever stops being true
    the move above is known to be unnecessary."""
    write_citation(dataset)
    described = _described(dataset)
    described["Authors"] = ["Karel Lopez"]
    (dataset / "dataset_description.json").write_text(json.dumps(described))

    report = validate(dataset)
    found = [
        i.rule_id for v in report.files for i in (v.issues or [])
        if i.severity is Severity.ERR
    ] + [
        i.rule_id for i in (report.dataset_issues or [])
        if i.severity is Severity.ERR
    ]
    assert any("MUTUALLY_EXCLUSIVE" in r for r in found), found


def test_what_will_move_can_be_asked_before_doing_it(dataset: Path) -> None:
    """A repair that removes something a user typed has to say so first."""
    assert fields_moved_by_citation(dataset) == ["Authors"]
    write_citation(dataset)
    assert fields_moved_by_citation(dataset) == [], "nothing left to move"


def test_a_description_with_no_authors_loses_nothing(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    described = {"Name": "d", "BIDSVersion": "1.10.0", "License": "CC0"}
    (root / "dataset_description.json").write_text(json.dumps(described))
    write_citation(root)
    assert _described(root) == described


def test_the_moved_set_is_exactly_one_field() -> None:
    """A guard on the constant itself: it grew to four once, and that is what
    deleted a user's licence."""
    assert MOVED_FIELDS == ("Authors",)
    assert set(MOVED_FIELDS) & set(COPIED_FIELDS) == set()


def test_the_split_matches_what_the_schema_asks_for() -> None:
    """Read the two rules out of the schema rather than trusting the reading.

    The whole split rests on a claim about the standard: that duplicating
    ``Authors`` is an error and duplicating the other three is a warning. That
    is checkable, and a claim about the standard that nobody checks is how the
    four-field version got written in the first place.
    """
    from bidsschematools import schema as bst

    rules = bst.load_schema().rules.checks.dataset

    authors = rules["SingleSourceAuthors"]
    assert authors["issue"]["level"] == "error"
    assert authors["issue"]["code"] == (
        "AUTHORS_AND_CITATION_FILE_MUTUALLY_EXCLUSIVE"
    )
    # The checks are "!(<Field> in dataset.dataset_description)" per field.
    assert [
        field for field in MOVED_FIELDS
        if any(field in check for check in authors["checks"])
    ] == list(MOVED_FIELDS)

    others = rules["SingleSourceCitationFields"]
    assert others["issue"]["level"] == "warning"
    assert sorted(COPIED_FIELDS) == sorted(
        field for field in COPIED_FIELDS
        if any(field in check for check in others["checks"])
    )
    # And the two rules do not overlap, which is what makes the split a split.
    assert len(others["checks"]) == len(COPIED_FIELDS)


def test_keeping_the_copied_fields_is_only_a_warning(dataset: Path) -> None:
    """The cost of the choice, stated as a test rather than as a comment.

    Keeping License, HowToAcknowledge and ReferencesAndLinks in both files
    raises one warning. That is the deliberate trade for not deleting a value
    the user typed, and if it ever becomes an error this test says so.
    """
    write_citation(dataset)
    report = validate(dataset)
    everything = [
        i for v in report.files for i in (v.issues or []) if not i.mirrored
    ] + list(report.dataset_issues or [])
    single_source = [
        i for i in everything if i.rule_id == "SINGLE_SOURCE_CITATION_FIELDS"
    ]
    assert single_source, "expected the validator to note the duplication"
    assert all(i.severity is not Severity.ERR for i in single_source)
