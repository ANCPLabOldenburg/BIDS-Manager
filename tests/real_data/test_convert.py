"""Real-data integration test for ``bidsmgr-convert``.

Runs the full scan + convert pipeline against a small fmap+func+anat
MRI dataset and asserts the BIDS layout, IntendedFor URIs, and
provenance look right. Gated on ``BIDS_MANAGER_REAL_MRI_DATA=1``.

Reference: super_plan.md §14.5 — real-data acceptance gate for the
converter loop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bidsmgr.cli.convert import run_convert
from bidsmgr.cli.scan import run_scan


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
def neuroimaging_unit_new_artifacts(tmp_path_factory) -> tuple[Path, Path]:
    """Scan once, convert once; return (bids_parent, dataset_root)."""
    work = tmp_path_factory.mktemp("neuroimaging_unit_new")
    tsv = work / "inventory.tsv"
    bids_parent = work / "out"

    run_scan(
        REAL_MRI_ROOT / "neuroimaging_unit_new",
        tsv,
        n_jobs=4,
    )
    rc = run_convert(tsv, bids_parent, n_jobs=4)
    assert rc == 0
    return bids_parent, bids_parent / "neuroimaging_unit_new"


class TestConvertNeuroimagingUnitNew:
    def test_dataset_root_with_description(
        self, neuroimaging_unit_new_artifacts,
    ) -> None:
        _, dataset_root = neuroimaging_unit_new_artifacts
        dd_path = dataset_root / "dataset_description.json"
        assert dd_path.exists()
        dd = json.loads(dd_path.read_text())
        assert dd["Name"] == "neuroimaging_unit_new"
        assert dd["DatasetType"] == "raw"
        assert any(
            entry.get("Name") == "bidsmgr" for entry in dd.get("GeneratedBy", [])
        )

    def test_subject_session_layout(
        self, neuroimaging_unit_new_artifacts,
    ) -> None:
        _, dataset_root = neuroimaging_unit_new_artifacts
        # sub-001 has two sessions (ses-pre, ses-post).
        sub1 = dataset_root / "sub-001"
        assert sub1.is_dir()
        assert (sub1 / "ses-pre" / "anat").is_dir()
        assert (sub1 / "ses-pre" / "fmap").is_dir()
        assert (sub1 / "ses-pre" / "func").is_dir()
        assert (sub1 / "ses-post" / "fmap").is_dir()
        assert (sub1 / "ses-post" / "func").is_dir()

    def test_fmap_outputs_have_bids_suffixes(
        self, neuroimaging_unit_new_artifacts,
    ) -> None:
        """No raw dcm2niix tokens (_e1/_e2/_ph) survive into the BIDS tree."""
        _, dataset_root = neuroimaging_unit_new_artifacts
        for fmap_file in dataset_root.rglob("fmap/*.nii.gz"):
            name = fmap_file.name
            assert "_e1." not in name and not name.endswith("_e1.nii.gz")
            assert "_e2." not in name and not name.endswith("_e2.nii.gz")
            assert "_ph." not in name and not name.endswith("_ph.nii.gz")
            # Every fmap NIfTI must have a recognised BIDS suffix.
            assert any(
                name.endswith(f"_{suf}.nii.gz")
                for suf in ("magnitude1", "magnitude2", "phasediff",
                            "phase1", "phase2", "fieldmap", "epi")
            ), f"no recognised fmap suffix in {name}"

    def test_intended_for_uses_bids_uri(
        self, neuroimaging_unit_new_artifacts,
    ) -> None:
        _, dataset_root = neuroimaging_unit_new_artifacts
        # Find any fmap JSON with IntendedFor populated.
        found_with_intended_for = False
        for sidecar in dataset_root.rglob("fmap/*.json"):
            data = json.loads(sidecar.read_text())
            intended = data.get("IntendedFor", [])
            if not intended:
                continue
            found_with_intended_for = True
            for entry in intended:
                assert entry.startswith("bids::sub-"), entry
                assert entry.endswith(".nii.gz"), entry
                # Session-scoped entries match the fmap's session.
                if "/ses-" in str(sidecar):
                    ses = str(sidecar).split("/ses-")[1].split("/", 1)[0]
                    assert f"/ses-{ses}/" in entry, (
                        f"fmap in ses-{ses} bound to a func URI in a "
                        f"different session: {entry}"
                    )
        assert found_with_intended_for, (
            "no fmap JSONs had IntendedFor populated — the dataset "
            "should produce at least one fmap+func pairing"
        )

    def test_provenance_per_subject(
        self, neuroimaging_unit_new_artifacts,
    ) -> None:
        _, dataset_root = neuroimaging_unit_new_artifacts
        for sub_dir in sorted(dataset_root.glob("sub-*")):
            prov_path = sub_dir / ".bidsmgr" / "provenance.json"
            assert prov_path.exists(), f"missing {prov_path}"
            prov = json.loads(prov_path.read_text())
            assert prov["bidsmgr_version"]
            assert "dcm2niix" in prov["dcm2niix_version"].lower()
            assert prov["tasks"], f"no tasks recorded for {sub_dir.name}"
            # At least one task should have succeeded.
            assert any(t["success"] for t in prov["tasks"])

    def test_no_staging_left_behind(
        self, neuroimaging_unit_new_artifacts,
    ) -> None:
        _, dataset_root = neuroimaging_unit_new_artifacts
        assert not (dataset_root / ".tmp_bidsmgr").exists()

    def test_dwi_emits_bval_bvec(
        self, neuroimaging_unit_new_artifacts,
    ) -> None:
        """sub-002 has DWI data — verify .bval/.bvec landed alongside .nii.gz."""
        _, dataset_root = neuroimaging_unit_new_artifacts
        dwi_niftis = list((dataset_root / "sub-002" / "dwi").glob("*.nii.gz"))
        if not dwi_niftis:
            pytest.skip("no DWI data in this fixture run")
        for nii in dwi_niftis:
            stem = nii.name[: -len(".nii.gz")]
            parent = nii.parent
            assert (parent / f"{stem}.bval").exists(), f"missing bval for {nii.name}"
            assert (parent / f"{stem}.bvec").exists(), f"missing bvec for {nii.name}"
            assert (parent / f"{stem}.json").exists(), f"missing json for {nii.name}"

    def test_physio_outputs_via_bidsphysio(
        self, neuroimaging_unit_new_artifacts,
    ) -> None:
        """Siemens CMRR ``_PhysioLog.dcm`` rows produce ``_physio.tsv.gz`` +
        ``_physio.json`` pairs in the same dir as the imaging data they
        accompany. Drives the ``PhysioDcmBackend`` end-to-end through
        the per-task dispatcher."""
        import json as _json

        _, dataset_root = neuroimaging_unit_new_artifacts
        physio_niftis = list(dataset_root.rglob("*_physio.tsv.gz"))
        if not physio_niftis:
            pytest.skip("no physio rows in this fixture run")
        for tsv in physio_niftis:
            stem = tsv.name[: -len(".tsv.gz")]
            sidecar = tsv.parent / f"{stem}.json"
            assert sidecar.exists(), f"missing JSON sidecar for {tsv.name}"
            data = _json.loads(sidecar.read_text())
            # BIDS spec: physio JSON must declare these three keys.
            assert "SamplingFrequency" in data, f"{sidecar.name}: missing SamplingFrequency"
            assert "StartTime" in data, f"{sidecar.name}: missing StartTime"
            assert "Columns" in data, f"{sidecar.name}: missing Columns"
            assert isinstance(data["Columns"], list)
            assert data["SamplingFrequency"] > 0
            # Physio outputs land in func/ (or beh/ etc.) of the
            # accompanying imaging data, not in a separate top-level dir.
            assert tsv.parent.name in {"func", "dwi", "anat", "perf"}, (
                f"unexpected physio location: {tsv.parent}"
            )
