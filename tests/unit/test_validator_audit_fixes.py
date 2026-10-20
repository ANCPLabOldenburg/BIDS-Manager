"""The vendored validator after the 1.5.1 audit.

Measured against the reference validator (3.0.2) on 61 single-defect copies
of a BIDS Manager dataset: what it missed (JSON formats, coordsystem files,
gzipped tables, header checks on single files), what it reported wrongly
(root-level gradient tables, findings piled on a file that is not BIDS,
derivative datasets, a duplicate subject warning), and how slow it was on a
large folder. One test per fix.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from bidsmgr.vendor.bidsval import validate


def _write(path: Path, content=b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content))
    elif isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def _dataset(root: Path, **description) -> Path:
    _write(root / "dataset_description.json",
           {"Name": "d", "BIDSVersion": "1.10.0", **description})
    _write(root / "participants.tsv", "participant_id\nsub-01\n")
    return root


def _codes(root: Path, *, deep: bool = False, where: str = "") -> list[str]:
    rep = validate(root, read_headers=deep)
    out = [i.code for i in rep.dataset_issues.issues]
    for v in rep.files:
        if where in str(v.path):
            out += [i.code for i in v.issues
                    if str(getattr(i.severity, "value", i.severity)) in ("error", "warning")]
    return out


def test_a_placeholder_in_a_time_field_is_an_error(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    pet = root / "sub-01" / "pet"
    _write(pet / "sub-01_pet.nii.gz", b"\x1f\x8b")
    _write(pet / "sub-01_pet.json", {"SpecificRadioactivityMeasTime": "TODO"})
    assert "JSON_SCHEMA_VALIDATION_ERROR" in _codes(root, where="sub-01_pet.json")
    _write(pet / "sub-01_pet.json", {"SpecificRadioactivityMeasTime": "09:28:23"})
    assert "JSON_SCHEMA_VALIDATION_ERROR" not in _codes(root, where="sub-01_pet.json")


def test_the_metadata_step_writes_no_placeholder_a_format_refuses() -> None:
    from bidsmgr.metadata.engine import NO_TODO, todo_value_for, unmarkable_reason
    from bidsmgr.schema import sidecar_fields

    field = next(f for f in sidecar_fields("pet", "pet")
                 if f.name == "SpecificRadioactivityMeasTime")
    assert field.format == "time" and not field.accepts_free_text
    assert todo_value_for(field) is NO_TODO
    assert unmarkable_reason(field) == "must be written as a time, which a placeholder is not"


def test_a_coordsystem_file_is_checked(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    meg = root / "sub-01" / "meg"
    _write(meg / "sub-01_task-rest_meg.fif", b"x")
    _write(meg / "sub-01_coordsystem.json", {"MEGCoordinateUnits": "m"})
    assert "JSON_KEY_REQUIRED" in _codes(root, where="coordsystem")


def test_a_gzipped_table_is_column_checked_by_its_sidecar(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    beh = root / "sub-01" / "beh"
    name = "sub-01_task-rest_recording-eye1_physio"
    _write(beh / f"{name}.json", {
        "PhysioType": "eyetrack", "SamplingFrequency": 1000, "StartTime": 0,
        "Columns": ["timestamp", "y_coordinate", "x_coordinate"]})
    _write(beh / f"{name}.tsv.gz", gzip.compress(b"1\t2\t3\n2\t2\t3\n"))
    assert "TSV_COLUMN_ORDER_INCORRECT" in _codes(root, where=name)
    _write(beh / f"{name}.json", {
        "PhysioType": "eyetrack", "SamplingFrequency": 1000, "StartTime": 0,
        "Columns": ["timestamp", "x_coordinate", "y_coordinate"]})
    assert "TSV_COLUMN_ORDER_INCORRECT" not in _codes(root, where=name)


def test_a_bids_uri_naming_nothing_is_reported(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    fmap = root / "sub-01" / "fmap"
    _write(root / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz", b"x")
    _write(fmap / "sub-01_phasediff.nii.gz", b"x")
    _write(fmap / "sub-01_phasediff.json", {
        "EchoTime1": 0.004, "EchoTime2": 0.006,
        "IntendedFor": ["bids::sub-01/func/sub-01_task-x_bold.nii.gz"]})
    assert "INTENDED_FOR" not in _codes(root, where="phasediff")
    _write(fmap / "sub-01_phasediff.json", {
        "EchoTime1": 0.004, "EchoTime2": 0.006,
        "IntendedFor": ["bids::sub-01/func/sub-01_task-gone_bold.nii.gz"]})
    assert "INTENDED_FOR" in _codes(root, where="phasediff")


def test_a_byte_order_mark_is_named_as_such(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    anat = root / "sub-01" / "anat"
    _write(anat / "sub-01_T1w.nii.gz", b"x")
    _write(anat / "sub-01_T1w.json", b"\xef\xbb\xbf{}")
    codes = _codes(root, where="T1w.json")
    assert "INVALID_JSON_ENCODING" in codes and "JSON_INVALID" not in codes


def test_root_gradient_tables_are_inherited_not_misplaced(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    _write(root / "sub-01" / "dwi" / "sub-01_dwi.nii.gz", b"x")
    _write(root / "dwi.bval", "0 1000\n")
    _write(root / "dwi.bvec", "0 1\n0 0\n0 0\n")
    assert "INVALID_LOCATION" not in _codes(root, where="dwi.bv")


def test_a_file_that_is_not_bids_gets_one_finding_not_twenty(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    _write(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz", b"x")
    _write(root / "sub-01" / "func" / "sub-01_task-x_boldd.nii.gz", b"x")
    codes = _codes(root, where="boldd")
    assert "NOT_INCLUDED" in codes
    assert not [c for c in codes if c.startswith("SIDECAR_KEY")]


def test_raw_sidecar_rules_are_not_demanded_of_a_derivative(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds", DatasetType="derivative",
                    GeneratedBy=[{"Name": "pipeline"}])
    _write(root / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz", b"x")
    _write(root / "sub-01" / "func" / "sub-01_task-x_bold.json", {"TaskName": "x"})
    rep = validate(root, read_headers=False)
    rules = {i.rule for v in rep.files for i in v.issues if i.code.startswith("SIDECAR_KEY")}
    assert rules, "the derivative rules still apply (SkullStripped)"
    assert all(r.startswith("rules.sidecars.derivatives.") for r in rules), rules


def test_no_subjects_is_said_once(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _write(root / "dataset_description.json", {"Name": "d", "BIDSVersion": "1.10.0"})
    codes = _codes(root)
    assert codes.count("SUBJECT_FOLDERS") == 1 and "NO_SUBJECTS" not in codes


def test_a_parsed_name_can_be_changed_by_its_caller(tmp_path: Path) -> None:
    """Parsing is memoised; every caller still gets its own entities."""
    from bidsmgr.schema import get_schema
    from bidsmgr.vendor.bidsval.context.entities import parse_filename

    schema = get_schema()
    first, _s, _e = parse_filename(schema, "sub-01_task-x_bold.nii.gz")
    first["sub"] = "changed"
    again, suffix, ext = parse_filename(schema, "sub-01_task-x_bold.nii.gz")
    assert again == {"sub": "01", "task": "x"} and (suffix, ext) == ("bold", ".nii.gz")


def test_the_editor_reads_headers_for_one_file_when_asked(tmp_path: Path) -> None:
    """Deep checks on Validate file: a 3-D BOLD is reported only when the
    header is read, and it used to be read only for the whole dataset."""
    nib = pytest.importorskip("nibabel")
    import numpy as np

    from bidsmgr.editor.validator import validate_file

    root = _dataset(tmp_path / "ds")
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)),
             func / "sub-01_task-x_bold.nii.gz")
    _write(func / "sub-01_task-x_bold.json", {"TaskName": "x", "RepetitionTime": 2.0})
    target = func / "sub-01_task-x_bold.nii.gz"
    fast = {i.rule_id for i in validate_file(root, target, strict=False).issues}
    deep = {i.rule_id for i in validate_file(root, target, strict=True).issues}
    assert "BOLD_NOT_4D" in deep and "BOLD_NOT_4D" not in fast
