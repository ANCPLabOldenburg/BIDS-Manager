"""Validating a dataset must not change it.

The report: "when I click validate dataset, I see new metadata fields appear
in the sidecars. Is this normal?"

It is, and nothing is written. What appears is the FORM offering fields: an
unvalidated file has no known datatype, so the form can only show the keys the
file holds; a validated one knows it is ``func/bold``, so the form can offer
every field the standard declares for that, ready to fill in. A declared field
nobody types into never reaches the file.

That distinction is only reassuring if it is true, so it is asserted here
rather than explained: validating, at either depth, leaves every byte of the
dataset alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bidsmgr.editor.validator import validate, validate_file, validate_folder


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    folder = root / "sub-01" / "func"
    folder.mkdir(parents=True)
    (folder / "sub-01_task-rest_bold.nii.gz").write_bytes(b"\0" * 64)
    (folder / "sub-01_task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest"}, indent=2)
    )
    (folder / "sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\n1.0\t0.5\n"
    )
    (root / "participants.tsv").write_text("participant_id\nsub-01\n")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}, indent=2)
    )
    return root


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha1(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".bidsmgr" not in path.parts
    }


@pytest.mark.parametrize("deep", [False, True])
def test_validating_a_dataset_changes_no_file(dataset: Path, deep: bool) -> None:
    before = _fingerprint(dataset)
    validate(dataset, strict=deep)
    assert _fingerprint(dataset) == before


def test_validating_creates_no_file(dataset: Path) -> None:
    before = set(_fingerprint(dataset))
    validate(dataset)
    assert set(_fingerprint(dataset)) == before


def test_validating_one_file_changes_nothing(dataset: Path) -> None:
    before = _fingerprint(dataset)
    validate_file(
        dataset, dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json",
    )
    assert _fingerprint(dataset) == before


def test_validating_a_folder_changes_nothing(dataset: Path) -> None:
    before = _fingerprint(dataset)
    validate_folder(dataset, dataset / "sub-01")
    assert _fingerprint(dataset) == before


def test_validating_twice_changes_nothing(dataset: Path) -> None:
    """Including any cache or report the run might want to leave behind."""
    validate(dataset)
    before = _fingerprint(dataset)
    validate(dataset)
    assert _fingerprint(dataset) == before


def test_the_sidecar_still_holds_exactly_what_it_held(dataset: Path) -> None:
    """Stated as the user would check it: open the file and read it."""
    sidecar = dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json"
    validate(dataset, strict=True)
    assert json.loads(sidecar.read_text()) == {"TaskName": "rest"}


def test_the_form_offers_more_than_the_file_holds(dataset: Path) -> None:
    """The thing that LOOKS like fields appearing: the verdict carries the
    declared fields so a form can offer them. They are a description of the
    standard, not a change to the file."""
    report = validate(dataset)
    verdict = next(
        v for v in report.files
        if Path(v.path).name == "sub-01_task-rest_bold.json"
    )
    offered = {field.name for field in verdict.sidecar_fields}
    assert len(offered) > 1, "the form has something to offer"
    assert any(not field.present for field in verdict.sidecar_fields)
    on_disk = set(json.loads(
        (dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json").read_text()
    ))
    assert on_disk == {"TaskName"}, "and the file is untouched"
