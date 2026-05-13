"""Real-data integration test for ``bidsmgr-metadata``.

Runs the full scan + convert + metadata pipeline against
``neuroimaging_unit_new`` (smallest fmap+func+anat dataset) and asserts
the dataset-level files land correctly. Gated on
``BIDS_MANAGER_REAL_MRI_DATA=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli.convert import run_convert
from bidsmgr.cli.scan import run_scan
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
def metadata_artifacts(tmp_path_factory) -> tuple[Path, Path, "MetadataReport"]:
    """Scan + convert + metadata once. Returns (bids_root, inventory_tsv, report)."""
    work = tmp_path_factory.mktemp("neuroimaging_unit_new_metadata")
    tsv = work / "inventory.tsv"
    bids_parent = work / "out"

    run_scan(REAL_MRI_ROOT / "neuroimaging_unit_new", tsv, n_jobs=4)
    rc = run_convert(tsv, bids_parent, n_jobs=4)
    assert rc == 0
    bids_root = bids_parent / "neuroimaging_unit_new"

    meta = DatasetMetadata(
        name="Neuroimaging Unit (test)",
        license="CC0",
        authors=["Karel Lopez"],
    )
    report = run_metadata(bids_root, inventory_tsv=tsv, dataset_meta=meta)
    return bids_root, tsv, report


class TestMetadataNeuroimagingUnitNew:
    def test_dataset_description_merged(self, metadata_artifacts) -> None:
        bids_root, _, _ = metadata_artifacts
        dd = json.loads((bids_root / "dataset_description.json").read_text())
        assert dd["Name"] == "Neuroimaging Unit (test)"
        assert dd["License"] == "CC0"
        assert dd["Authors"] == ["Karel Lopez"]
        # GeneratedBy chain: converter entry + metadata entry, both bidsmgr.
        names = [e.get("Name") for e in dd["GeneratedBy"]]
        assert "bidsmgr" in names
        descs = [e.get("Description", "") for e in dd["GeneratedBy"]]
        assert any("dcm2niix" in d for d in descs)
        assert any("metadata" in d for d in descs)

    def test_participants_enriched_from_inventory(self, metadata_artifacts) -> None:
        bids_root, _, _ = metadata_artifacts
        df = pd.read_csv(
            bids_root / "participants.tsv",
            sep="\t", dtype=str, keep_default_na=False,
        )
        assert "participant_id" in df.columns
        # Should have at least the demographic columns coming from the
        # bidsmgr-shaped inventory (PatientID/PatientAge/PatientSex/...).
        for col in ("age", "sex", "patient_id"):
            assert col in df.columns, f"missing column: {col}"
        assert len(df) >= 1
        # No participant has every column at "n/a" (the inventory has
        # real PatientID values).
        assert any(df.iloc[i]["patient_id"] != "n/a" for i in range(len(df)))

    def test_readme_and_changes_seeded(self, metadata_artifacts) -> None:
        bids_root, _, _ = metadata_artifacts
        assert (bids_root / "README").exists()
        assert (bids_root / "CHANGES").exists()
        assert (bids_root / "README").read_text().startswith("# Neuroimaging Unit (test)")

    def test_scans_tsv_per_session(self, metadata_artifacts) -> None:
        bids_root, _, _ = metadata_artifacts
        # neuroimaging_unit_new has sessions for sub-001.
        scans_files = list(bids_root.rglob("*_scans.tsv"))
        assert scans_files, "expected at least one *_scans.tsv"
        for scans in scans_files:
            df = pd.read_csv(scans, sep="\t", dtype=str, keep_default_na=False)
            assert list(df.columns) == ["filename", "acq_time"]
            assert len(df) > 0

    def test_taskname_filled_from_filename(self, metadata_artifacts) -> None:
        bids_root, _, _ = metadata_artifacts
        bold_jsons = list(bids_root.rglob("func/*_bold.json"))
        assert bold_jsons, "expected at least one bold sidecar"
        for j in bold_jsons:
            data = json.loads(j.read_text())
            # TaskName should match the _task-<X> token in the filename.
            assert "TaskName" in data, f"missing TaskName in {j.name}"
            # Quick sanity: no junk like an empty string.
            assert data["TaskName"]

    def test_audit_does_not_flag_volumetiming_when_repetition_time_present(
        self, metadata_artifacts,
    ) -> None:
        _, _, report = metadata_artifacts
        # The fixture's bold sidecars carry RepetitionTime → VolumeTiming
        # MUST NOT appear in missing_required (mutual exclusivity).
        assert all(
            "VolumeTiming" not in msg for msg in report.missing_required
        ), [m for m in report.missing_required if "VolumeTiming" in m]

    def test_audit_field_names_canonical(self, metadata_artifacts) -> None:
        _, _, report = metadata_artifacts
        # No bidsschematools rule-decorated names in user-facing messages.
        for msg in report.missing_required + report.missing_recommended:
            assert "__" not in msg, msg
