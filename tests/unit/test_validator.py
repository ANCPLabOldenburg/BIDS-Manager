"""Unit tests for ``bidsmgr.editor.validator``.

Synthetic BIDS trees, no real DICOMs, no dcm2niix. Each test builds a
minimal directory shape with hand-written JSON and asserts on the
:class:`ValidationReport` Pydantic structure that the GUI will consume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import (
    FieldLevel,
    Severity,
    ValidationReport,
    rollup_severity,
    validate,
)
from bidsmgr.editor.validator import _resolve_intended_for, _value_kind


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_pair(folder: Path, basename: str, *, sidecar: dict | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{basename}.nii.gz").write_bytes(b"\x1f\x8b")
    (folder / f"{basename}.json").write_text(json.dumps(sidecar or {}, indent=2))


def _make_dd(root: Path, *, name: str | None = None, extra: dict | None = None) -> Path:
    """Write a minimal valid dataset_description.json."""
    body = {
        "Name": name or root.name,
        "BIDSVersion": "1.10.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "bidsmgr", "Version": "0.0.1"}],
    }
    if extra:
        body.update(extra)
    p = root / "dataset_description.json"
    p.write_text(json.dumps(body, indent=2))
    return p


def _make_minimal_bids(tmp_path: Path) -> Path:
    root = tmp_path / "study"
    root.mkdir()
    _make_dd(root)

    sub = root / "sub-001"
    _write_pair(sub / "anat", "sub-001_T1w",
                sidecar={"AcquisitionTime": "120000"})
    _write_pair(sub / "func", "sub-001_task-rest_bold",
                sidecar={"AcquisitionTime": "120500",
                         "TaskName": "rest",
                         "RepetitionTime": 2.0})
    return root


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------


class TestValidateBasics:
    def test_returns_pydantic_report(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        report = validate(root)
        assert isinstance(report, ValidationReport)
        assert report.bidsmgr_version
        assert report.bids_version
        assert report.generated_at
        assert report.bids_root == root.resolve()

    def test_minimal_dataset_has_no_errors(self, tmp_path: Path) -> None:
        """A minimal valid dataset has no ERR-level issues. Some
        recommended fields are still missing (no real DICOM-derived
        metadata to fill), so severity is WARN, not OK — that's the
        expected behavior on a hand-built fixture.
        """
        root = _make_minimal_bids(tmp_path)
        report = validate(root)
        assert report.counts["err"] == 0
        # Required-field violations should be zero on this fixture.
        for f in report.files:
            assert all(
                i.rule_id != "bids.required_sidecar_field_missing"
                for i in f.issues
            ), [(i.rule_id, i.message) for i in f.issues]

    def test_raises_when_root_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate(tmp_path / "no-such-dir")

    def test_files_use_relative_paths(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        report = validate(root)
        for f in report.files:
            # Path is relative to bids_root, never absolute.
            assert not f.path.is_absolute()


# ---------------------------------------------------------------------------
# Dataset-root checks
# ---------------------------------------------------------------------------


class TestDatasetRoot:
    def test_missing_dataset_description_is_error(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        (root / "sub-001" / "anat").mkdir(parents=True)
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        report = validate(root)
        assert report.severity is Severity.ERR
        assert any(
            i.rule_id == "bids.missing_dataset_description"
            for i in report.dataset_issues
        )

    def test_invalid_json_dataset_description(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        (root / "dataset_description.json").write_text("{not json")
        (root / "sub-001" / "anat").mkdir(parents=True)
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        report = validate(root)
        assert report.severity is Severity.ERR
        dd_verdict = next(f for f in report.files if f.path.name == "dataset_description.json")
        assert dd_verdict.severity is Severity.ERR

    def test_dataset_description_missing_required_field(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        (root / "dataset_description.json").write_text(
            json.dumps({"Name": "x"})  # no BIDSVersion
        )
        (root / "sub-001" / "anat").mkdir(parents=True)
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        report = validate(root)
        dd = next(f for f in report.files if f.path.name == "dataset_description.json")
        assert dd.severity is Severity.ERR
        assert any(i.field == "BIDSVersion" for i in dd.issues)

    def test_dataset_description_todo_placeholder_is_warning(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root, extra={"License": "TODO"})
        (root / "sub-001" / "anat").mkdir(parents=True)
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        report = validate(root)
        dd = next(f for f in report.files if f.path.name == "dataset_description.json")
        assert dd.severity is Severity.WARN
        assert any(
            i.rule_id == "bidsmgr.todo_placeholder" and i.field == "License"
            for i in dd.issues
        )

    def test_no_subjects_emits_dataset_warning(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        report = validate(root)
        assert any(
            i.rule_id == "bidsmgr.no_subjects" for i in report.dataset_issues
        )


# ---------------------------------------------------------------------------
# Per-file sidecar audit + sidecar_fields population
# ---------------------------------------------------------------------------


class TestSidecarAudit:
    def test_missing_required_field_is_error(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        # bold without RepetitionTime → schema flags as error.
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest"})  # no RepetitionTime
        report = validate(root)
        bold = next(
            f for f in report.files if f.path.name.endswith("_bold.json")
        )
        assert bold.severity is Severity.ERR
        assert any(
            i.rule_id == "bids.required_sidecar_field_missing"
            and i.field == "RepetitionTime"
            for i in bold.issues
        )

    def test_repetition_time_satisfies_volume_timing_alternative(
        self, tmp_path: Path,
    ) -> None:
        """Mutual-exclusivity: RepetitionTime present → no VolumeTiming flag."""
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0})
        report = validate(root)
        bold = next(
            f for f in report.files if f.path.name.endswith("_bold.json")
        )
        assert all(
            i.field != "VolumeTiming" for i in bold.issues
        ), [(i.field, i.message) for i in bold.issues]

    def test_canonical_field_names_in_messages(self, tmp_path: Path) -> None:
        """fmap/phase2 wants ``EchoTime`` (the schema reports it as
        ``EchoTime__fmap``). User-facing message must use the clean name.
        """
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "fmap", "sub-001_phase2", sidecar={})
        report = validate(root)
        phase = next(
            f for f in report.files if f.path.name.endswith("_phase2.json")
        )
        # No rule-decorated names anywhere.
        for issue in phase.issues:
            assert "__" not in issue.message
            if issue.field:
                assert "__" not in issue.field

    def test_recommended_missing_is_warning(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        # dwi has recommended fields but no required ones.
        _write_pair(root / "sub-001" / "dwi", "sub-001_dwi", sidecar={})
        report = validate(root)
        dwi = next(f for f in report.files if f.path.name.endswith("_dwi.json"))
        assert dwi.severity is Severity.WARN
        assert any(
            i.rule_id == "bids.recommended_sidecar_field_missing"
            for i in dwi.issues
        )

    def test_sidecar_fields_populated_with_levels(self, tmp_path: Path) -> None:
        """Every schema-defined field appears as a SidecarField row with
        the right level."""
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0,
                             "MyCustom": 42})
        report = validate(root)
        bold = next(
            f for f in report.files if f.path.name.endswith("_bold.json")
        )
        names = {sf.name: sf for sf in bold.sidecar_fields}
        # Required fields present → level=REQUIRED, present=True.
        assert names["TaskName"].level is FieldLevel.REQUIRED
        assert names["TaskName"].present is True
        assert names["TaskName"].value == "rest"
        # Custom keys appear as OPTIONAL.
        assert names["MyCustom"].level is FieldLevel.OPTIONAL
        assert names["MyCustom"].value_kind == "number"
        # Recommended fields that are missing still appear as rows.
        assert "Instructions" in names
        assert names["Instructions"].present is False
        assert names["Instructions"].value_kind == "missing"


class TestTodoPlaceholders:
    def test_todo_in_sidecar_is_warning(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0,
                             "Instructions": "TODO"})
        report = validate(root)
        bold = next(
            f for f in report.files if f.path.name.endswith("_bold.json")
        )
        assert bold.severity is Severity.WARN
        todo_issues = [
            i for i in bold.issues if i.rule_id == "bidsmgr.todo_placeholder"
        ]
        assert len(todo_issues) == 1
        assert todo_issues[0].field == "Instructions"


class TestIntendedForResolution:
    def test_unresolved_uri_is_error(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "fmap", "sub-001_phasediff",
                    sidecar={
                        "EchoTime1": 0.005,
                        "EchoTime2": 0.007,
                        "IntendedFor": [
                            "bids::sub-001/func/does-not-exist_bold.nii.gz",
                        ],
                    })
        # add a func sibling so that fmap can theoretically point there
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0})
        report = validate(root)
        phasediff = next(
            f for f in report.files if f.path.name.endswith("_phasediff.json")
        )
        assert phasediff.severity is Severity.ERR
        assert any(
            i.rule_id == "bids.intended_for_unresolved" for i in phasediff.issues
        )

    def test_resolvable_uri_is_ok(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0})
        _write_pair(root / "sub-001" / "fmap", "sub-001_phasediff",
                    sidecar={
                        "EchoTime1": 0.005,
                        "EchoTime2": 0.007,
                        "IntendedFor": [
                            "bids::sub-001/func/sub-001_task-rest_bold.nii.gz",
                        ],
                    })
        report = validate(root)
        phasediff = next(
            f for f in report.files if f.path.name.endswith("_phasediff.json")
        )
        assert all(
            i.rule_id != "bids.intended_for_unresolved" for i in phasediff.issues
        )

    def test_resolve_helper_handles_uri_form(self, tmp_path: Path) -> None:
        bids = tmp_path / "study"
        sidecar = bids / "sub-1" / "fmap" / "sub-1_phasediff.json"
        target = bids / "sub-1" / "func" / "sub-1_task-rest_bold.nii.gz"
        result = _resolve_intended_for(
            "bids::sub-1/func/sub-1_task-rest_bold.nii.gz", sidecar, bids,
        )
        assert result == target

    def test_resolve_helper_handles_legacy_relative(self, tmp_path: Path) -> None:
        bids = tmp_path / "study"
        sidecar = bids / "sub-1" / "ses-pre" / "fmap" / "sub-1_ses-pre_phasediff.json"
        target = bids / "sub-1" / "ses-pre" / "func" / "sub-1_ses-pre_task-rest_bold.nii.gz"
        # Legacy: relative to subject root.
        result = _resolve_intended_for(
            "ses-pre/func/sub-1_ses-pre_task-rest_bold.nii.gz", sidecar, bids,
        )
        assert result == target


# ---------------------------------------------------------------------------
# Layout checks
# ---------------------------------------------------------------------------


class TestLayoutChecks:
    def test_unknown_datatype_dir_is_warning(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        # Standard subject + a stray "raw_dicoms" directory that isn't
        # a known BIDS datatype.
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        (root / "sub-001" / "raw_dicoms").mkdir()

        report = validate(root)
        assert "sub-001" in report.folder_issues
        assert any(
            i.rule_id == "bids.unknown_datatype_dir"
            for i in report.folder_issues["sub-001"]
        )

    def test_dot_directories_are_silent(self, tmp_path: Path) -> None:
        """Dot-prefixed dirs (.bidsmgr, .git) are not flagged as orphan."""
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        # Add bookkeeping dirs that the validator should ignore.
        (root / "sub-001" / ".bidsmgr").mkdir()
        (root / "sub-001" / ".git").mkdir()
        report = validate(root)
        assert "sub-001" not in report.folder_issues


# ---------------------------------------------------------------------------
# Severity rollup
# ---------------------------------------------------------------------------


class TestRollupSeverity:
    def test_empty_is_ok(self) -> None:
        assert rollup_severity([]) is Severity.OK

    def test_one_warn(self) -> None:
        assert rollup_severity([Severity.OK, Severity.WARN]) is Severity.WARN

    def test_err_dominates(self) -> None:
        assert (
            rollup_severity([Severity.OK, Severity.WARN, Severity.ERR])
            is Severity.ERR
        )


class TestValueKind:
    @pytest.mark.parametrize("value,expected", [
        (None, "null"),
        (True, "bool"),
        (False, "bool"),
        (42, "number"),
        (3.14, "number"),
        ("rest", "string"),
        ("TODO", "todo"),
        ([1, 2], "array"),
        ({"a": 1}, "object"),
    ])
    def test_classification(self, value, expected) -> None:
        assert _value_kind(value) == expected


# ---------------------------------------------------------------------------
# Layer 2 — bidsschematools structural
# ---------------------------------------------------------------------------


class TestStrictMode:
    def test_strict_runs_layer_2_without_raising(self, tmp_path: Path) -> None:
        """Layer 2 runs without crashing on a well-formed dataset."""
        root = _make_minimal_bids(tmp_path)
        report = validate(root, strict=True)
        # No bidsschematools-specific failures expected on a minimal tree.
        assert all(
            i.rule_id != "bidsmgr.bst_failed" for i in report.dataset_issues
        )

    def test_strict_off_does_not_emit_bst_issues(self, tmp_path: Path) -> None:
        """Without --strict, no bidsschematools-derived issue ids appear."""
        root = _make_minimal_bids(tmp_path)
        report = validate(root, strict=False)
        bst_rule_ids = {
            "bids.unknown_path",
            "bids.missing_mandatory_path",
            "bidsmgr.bst_unavailable",
            "bidsmgr.bst_failed",
        }
        for f in report.files:
            for i in f.issues:
                assert i.rule_id not in bst_rule_ids
        for issue_list in report.folder_issues.values():
            for i in issue_list:
                assert i.rule_id not in bst_rule_ids
        for i in report.dataset_issues:
            assert i.rule_id not in bst_rule_ids
