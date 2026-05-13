"""Real-data tests for the ``--probe-convert`` flow.

The probe pass:

* Always uses ``<output_tsv_parent>/.tmp/`` as its scratch tree.
* Always wipes that tree when ``run_scan`` returns — there is no flag
  to keep it. Debugging is done with the lower-level
  :func:`bidsmgr.inventory.probe_convert.probe_rows` against a pytest
  ``tmp_path``.
* Runs each detected sequence as its own dcm2niix invocation, in
  parallel (``-j`` / ``n_jobs``).

Output landing directory for the user-visible TSVs:
``/Users/karelo/Development/datasets/BIDS_Manager/bids_manager_outputs/testing/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli.scan import PROBE_COLUMNS, run_scan
from bidsmgr.inventory import probe_convert as probe_convert_module
from bidsmgr.inventory.mri_dicom import scan_dicoms_long

REAL_MRI_ROOT = Path(
    "/Users/karelo/Development/datasets/BIDS_Manager/raw_data/MRI"
)
TESTING_OUT_ROOT = Path(
    "/Users/karelo/Development/datasets/BIDS_Manager/bids_manager_outputs/testing"
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
def neuroimaging_unit_new_probe() -> pd.DataFrame:
    """Run probe-convert on the small Prisma dataset and return the TSV.

    The probe ``.tmp/`` directory is auto-cleaned by ``run_scan``; the
    inventory TSV stays in the user's testing root.
    """
    src = REAL_MRI_ROOT / "neuroimaging_unit_new"
    if not src.exists():
        pytest.skip("neuroimaging_unit_new missing")
    TESTING_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_tsv = TESTING_OUT_ROOT / "neuroimaging_unit_new.tsv"
    run_scan(src, out_tsv, n_jobs=4, probe_convert=True)
    return pd.read_csv(out_tsv, sep="\t", keep_default_na=False, dtype=str)


def test_probe_tmp_is_auto_deleted(tmp_path: Path):
    """The default ``<output_tsv_parent>/.tmp/`` is removed when
    ``run_scan`` returns. The inventory TSV remains."""
    src = REAL_MRI_ROOT / "neuroimaging_unit_new"
    if not src.exists():
        pytest.skip("neuroimaging_unit_new missing")
    out_tsv = tmp_path / "scan.tsv"
    run_scan(src, out_tsv, n_jobs=4, probe_convert=True)
    assert out_tsv.exists()
    assert not (tmp_path / ".tmp").exists(), (
        ".tmp/ must be auto-deleted; found it under the output parent"
    )


def test_probe_rows_low_level_layout_is_per_series(tmp_path: Path):
    """Direct call into ``probe_rows`` (debug entry point) lays files out
    as ``<work_root>/sub-XXX/<series_uid>/{_dicoms,out}/``. ``run_scan``
    wraps this and wipes ``work_root`` when done; tests use the
    low-level entry to inspect the layout against a pytest tmp_path
    that pytest cleans up itself.
    """
    src = REAL_MRI_ROOT / "neuroimaging_unit_new"
    if not src.exists():
        pytest.skip("neuroimaging_unit_new missing")
    df = scan_dicoms_long(src, n_jobs=4)
    files_by_uid = df.attrs["files_by_uid"]
    # Convert DataFrame rows back into InventoryRows the way cli.scan does.
    from bidsmgr.cli.scan import _rows_from_dataframe
    rows = _rows_from_dataframe(df)
    work_root = tmp_path / "probe"
    stats = probe_convert_module.probe_rows(
        rows, work_root, files_by_uid, n_jobs=4,
    )
    assert stats, "probe should have produced stats for at least one series"
    sub_buckets = [p for p in work_root.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    assert sub_buckets, f"no sub-*/ buckets in {work_root}"
    sample_series = next(
        (s for b in sub_buckets for s in b.iterdir() if s.is_dir()),
        None,
    )
    assert sample_series is not None
    assert (sample_series / "_dicoms").is_dir()
    assert (sample_series / "out").is_dir()


def test_probe_columns_present(neuroimaging_unit_new_probe: pd.DataFrame):
    for col in PROBE_COLUMNS:
        assert col in neuroimaging_unit_new_probe.columns, f"missing column: {col}"


def test_bold_rows_produce_one_nifti_in_per_series_mode(
    neuroimaging_unit_new_probe: pd.DataFrame,
):
    """Every func/bold row produces exactly 1 NIfTI when probed per-series.

    A ``probe_n_nifti > 1`` here would indicate a real intra-series
    split (e.g. operator-cancelled volume). Such cases must surface
    with a probe anomaly note in ``proposed_issues``.
    """
    bold_rows = neuroimaging_unit_new_probe[
        (neuroimaging_unit_new_probe["bids_guess_datatype"] == "func")
        & (neuroimaging_unit_new_probe["bids_guess_suffix"] == "bold")
        & (neuroimaging_unit_new_probe["probe_n_nifti"].astype(str) != "")
    ]
    assert len(bold_rows) >= 1
    for _, r in bold_rows.iterrows():
        n_nifti = int(r["probe_n_nifti"])
        if n_nifti != 1:
            assert "probe:" in str(r["proposed_issues"]), (
                f"bold row produced {n_nifti} NIfTI(s) in per-series mode "
                f"but proposed_issues doesn't record the anomaly: "
                f"{r['sequence']!r}"
            )
        assert int(r["probe_n_volumes"]) >= 1, (
            f"bold row should have at least 1 volume: {r['sequence']!r}"
        )


def test_fmap_rows_produce_three_niftis(neuroimaging_unit_new_probe: pd.DataFrame):
    """``gre_field_mapping`` produces magnitude1 + magnitude2 + phasediff."""
    fmap_rows = neuroimaging_unit_new_probe[
        (neuroimaging_unit_new_probe["bids_guess_datatype"] == "fmap")
        & (neuroimaging_unit_new_probe["bids_guess_suffix"].isin(
            ["magnitude1", "magnitude2", "phasediff"]
        ))
        & (neuroimaging_unit_new_probe["probe_n_nifti"].astype(str) != "")
    ]
    assert len(fmap_rows) >= 1
    for _, r in fmap_rows.iterrows():
        assert int(r["probe_n_nifti"]) >= 3, (
            f"fmap row {r['sequence']} produced only {r['probe_n_nifti']} NIfTI(s)"
        )


def test_dwi_rows_have_bval_bvec_in_extensions(neuroimaging_unit_new_probe: pd.DataFrame):
    """Raw DWI must produce .bval and .bvec sidecars."""
    dwi_rows = neuroimaging_unit_new_probe[
        (neuroimaging_unit_new_probe["bids_guess_datatype"] == "dwi")
        & (neuroimaging_unit_new_probe["bids_guess_suffix"] == "dwi")
        & (neuroimaging_unit_new_probe["probe_extensions"].astype(str) != "")
    ]
    assert len(dwi_rows) >= 1
    for _, r in dwi_rows.iterrows():
        exts = set(r["probe_extensions"].split(","))
        assert ".bval" in exts, f"dwi row missing .bval: {r['sequence']}"
        assert ".bvec" in exts, f"dwi row missing .bvec: {r['sequence']}"
        assert ".nii.gz" in exts, f"dwi row missing .nii.gz: {r['sequence']}"
