"""A finding belongs to the file you would edit to fix it.

bidsval attaches a metadata finding to the DATA file, because that is what the
standard describes: ``sub-01_bold.nii.gz`` is the thing that must have a
``RepetitionTime``. But the finding is not about the image, it is about the
sidecar, and the sidecar is where a user reads it, edits it and fixes it.
Counting it against the image put the number in the tree beside a file nothing
can be done to.

So a field-bearing finding is MOVED to the sibling ``.json``, and a mirror is
left on the data file. Findings about the data itself carry no field and stay
put. The totals do not change: only where they are counted does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor.types import Severity
from bidsmgr.editor.validator import validate


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    folder = root / "sub-01" / "func"
    folder.mkdir(parents=True)
    (folder / "sub-01_task-rest_bold.nii.gz").write_bytes(b"\0" * 16)
    (folder / "sub-01_task-rest_bold.json").write_text(
        json.dumps({"TaskName": "rest"})
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def _verdict(report, name: str):
    return next(v for v in report.files if Path(v.path).name == name)


def _counted(verdict) -> list:
    return [i for i in (verdict.issues or []) if not i.mirrored]


def _mirrored(verdict) -> list:
    return [i for i in (verdict.issues or []) if i.mirrored]


def test_metadata_findings_are_counted_on_the_sidecar(dataset: Path) -> None:
    report = validate(dataset)
    sidecar = _verdict(report, "sub-01_task-rest_bold.json")
    counted = _counted(sidecar)
    assert counted, "the sidecar should carry the metadata findings"
    assert all(i.field for i in counted)


def test_the_data_file_keeps_a_mirror_not_the_count(dataset: Path) -> None:
    """A reader looking at the image still sees that something is said about
    it, but the number in the tree sits where the work is."""
    report = validate(dataset)
    image = _verdict(report, "sub-01_task-rest_bold.nii.gz")
    assert not _counted(image), "nothing to do to the image itself"
    assert _mirrored(image), "but the findings are still visible there"


def test_no_finding_is_counted_twice(dataset: Path) -> None:
    report = validate(dataset)
    sidecar = _verdict(report, "sub-01_task-rest_bold.json")
    image = _verdict(report, "sub-01_task-rest_bold.nii.gz")
    fields_on_sidecar = {i.field for i in _counted(sidecar)}
    fields_on_image = {i.field for i in _counted(image)}
    assert not (fields_on_sidecar & fields_on_image)


def test_the_totals_are_unchanged_by_the_move(dataset: Path) -> None:
    """Moving where a finding is counted must not change how many there are."""
    report = validate(dataset)
    counted = [
        i for v in report.files for i in (v.issues or []) if not i.mirrored
    ]
    assert report.counts["warn"] == sum(
        1 for i in counted if i.severity is Severity.WARN
    )
    assert report.counts["err"] == sum(
        1 for i in counted if i.severity is Severity.ERR
    )


def test_a_finding_about_the_data_stays_on_the_data(tmp_path: Path) -> None:
    """An empty file is a fact about the file, not about its metadata, and
    carries no field. It must not be moved anywhere."""
    root = tmp_path / "ds"
    folder = root / "sub-01" / "anat"
    folder.mkdir(parents=True)
    (folder / "sub-01_T1w.nii.gz").write_bytes(b"")      # empty on purpose
    (folder / "sub-01_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    report = validate(root)
    image = _verdict(report, "sub-01_T1w.nii.gz")
    own = [i for i in _counted(image) if not i.field]
    assert own, "a fact about the file stays on the file"
    assert any("EMPTY" in (i.rule_id or "").upper() for i in own)


def test_a_data_file_with_no_sidecar_keeps_its_findings(
    tmp_path: Path,
) -> None:
    """Moving a finding to a file that does not exist would hide it."""
    root = tmp_path / "ds"
    folder = root / "sub-01" / "anat"
    folder.mkdir(parents=True)
    (folder / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 16)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    report = validate(root)
    names = {Path(v.path).name for v in report.files}
    assert "sub-01_T1w.json" not in names
    image = _verdict(report, "sub-01_T1w.nii.gz")
    assert _counted(image), "its findings must not have gone anywhere"
