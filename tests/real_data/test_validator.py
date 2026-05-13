"""Real-data integration test for ``bidsmgr.editor.validator``.

Runs scan + convert + metadata + validate on
``neuroimaging_unit_new`` and asserts the validator produces a
sensible :class:`ValidationReport` (no errors, expected warnings,
sidecar_fields populated).

Gated on ``BIDS_MANAGER_REAL_MRI_DATA=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bidsmgr.cli.convert import run_convert
from bidsmgr.cli.scan import run_scan
from bidsmgr.editor import FieldLevel, Severity, ValidationReport, validate
from bidsmgr.metadata import DatasetMetadata, run_metadata


REAL_MRI_ROOT = Path(
    "/Users/karelo/Development/datasets/BIDS_Manager/raw_data/MRI"
)


pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not os.environ.get("BIDS_MANAGER_REAL_MRI_DATA"),
        reason="BIDS_MANAGER_REAL_MRI_DATA not set",
    ),
    pytest.mark.skipif(
        not REAL_MRI_ROOT.exists(),
        reason=f"real MRI dataset root missing at {REAL_MRI_ROOT}",
    ),
]


@pytest.fixture(scope="module")
def validated_artifacts(tmp_path_factory) -> tuple[Path, ValidationReport]:
    """Scan + convert + metadata once; return (bids_root, report)."""
    work = tmp_path_factory.mktemp("neuroimaging_unit_new_validate")
    tsv = work / "inventory.tsv"
    bids_parent = work / "out"

    run_scan(REAL_MRI_ROOT / "neuroimaging_unit_new", tsv, n_jobs=4)
    rc = run_convert(tsv, bids_parent, n_jobs=4)
    assert rc == 0
    bids_root = bids_parent / "neuroimaging_unit_new"
    run_metadata(bids_root, inventory_tsv=tsv,
                 dataset_meta=DatasetMetadata(name="neuroimaging_unit_new"))

    report = validate(bids_root, strict=True)
    return bids_root, report


class TestValidateNeuroimagingUnitNew:
    def test_no_errors_on_a_freshly_converted_dataset(
        self, validated_artifacts,
    ) -> None:
        _, report = validated_artifacts
        assert report.counts["err"] == 0
        # No required-field violations expected after metadata pass.
        for f in report.files:
            assert all(
                i.rule_id != "bids.required_sidecar_field_missing"
                for i in f.issues
            ), [(i.rule_id, i.message) for i in f.issues]

    def test_severity_rolls_up_correctly(
        self, validated_artifacts,
    ) -> None:
        _, report = validated_artifacts
        # Recommended fields aren't all filled by DICOM → expect at
        # least one WARN somewhere → overall severity is WARN.
        assert report.severity is Severity.WARN
        assert report.counts["warn"] > 0

    def test_no_unresolved_intended_for_uris(
        self, validated_artifacts,
    ) -> None:
        _, report = validated_artifacts
        for f in report.files:
            assert all(
                i.rule_id != "bids.intended_for_unresolved" for i in f.issues
            ), f"{f.path}: {[i.message for i in f.issues]}"

    def test_sidecar_fields_have_expected_levels(
        self, validated_artifacts,
    ) -> None:
        """Every bold sidecar exposes TaskName as REQUIRED level."""
        _, report = validated_artifacts
        bold_verdicts = [
            f for f in report.files if f.path.name.endswith("_bold.json")
        ]
        assert bold_verdicts, "expected at least one bold sidecar"
        for v in bold_verdicts:
            taskname_rows = [
                sf for sf in v.sidecar_fields if sf.name == "TaskName"
            ]
            assert taskname_rows, f"TaskName row missing for {v.path}"
            assert taskname_rows[0].level is FieldLevel.REQUIRED
            assert taskname_rows[0].present is True

    def test_validation_report_json_written(
        self, validated_artifacts, tmp_path_factory,
    ) -> None:
        """Run via the CLI to confirm validation_report.json lands on disk."""
        bids_root, _ = validated_artifacts
        from bidsmgr.cli.validate import run_validate_cli

        rc = run_validate_cli(bids_root.parent, dataset=bids_root.name)
        # rc=1 on dataset-level WARN+ is expected; just confirm the side
        # effect (file written) regardless of exit code.
        assert rc in (0, 1)
        report_path = bids_root / ".bidsmgr" / "validation_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["bids_root"]
        assert data["bidsmgr_version"]
        assert isinstance(data["files"], list)

    def test_path_fields_are_relative(
        self, validated_artifacts,
    ) -> None:
        """All paths inside FileVerdicts are relative — the GUI binds them
        to a tree rooted at bids_root."""
        _, report = validated_artifacts
        for f in report.files:
            assert not f.path.is_absolute(), f.path
