"""Characterisation test: ``bidsmgr-scan`` works on every dataset under
``raw_data/MRI/``.

Gated on ``BIDS_MANAGER_REAL_MRI_DATA``. Iterates over every immediate
subdirectory of the MRI raw-data root and runs a scan against it, checking
only the *invariants*:

* The TSV is written and readable.
* The 22-column v0.2.5 contract is preserved.
* The 6 ``bids_guess_*`` columns are appended.
* Every populated ``proposed_basename`` validates against the BIDS schema.

This is intentionally lax about content (some datasets have only fMRI, some
have no T1, ``Old_LNF`` has just one EPI per subject, …). The contract being
tested is that the scanner produces a well-formed, schema-consistent TSV on
arbitrary real data — not that any specific datatype shows up.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr import schema as bids_schema
from bidsmgr.cli.scan import BIDS_GUESS_COLUMNS, run_scan
from bidsmgr.inventory.mri_dicom import TSV_COLUMNS

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


def _dataset_dirs() -> list[Path]:
    if not REAL_MRI_ROOT.exists():
        return []
    return sorted(p for p in REAL_MRI_ROOT.iterdir() if p.is_dir())


@pytest.mark.parametrize("dataset", _dataset_dirs(), ids=lambda p: p.name)
def test_scan_produces_valid_tsv(dataset: Path, tmp_path: Path):
    out = tmp_path / f"{dataset.name}.tsv"
    df = run_scan(dataset, out, n_jobs=4)

    assert out.exists(), "TSV not written"
    written = pd.read_csv(out, sep="\t", keep_default_na=False, dtype=str)

    # 22-column v0.2.5 contract.
    columns = list(written.columns)
    expected_22 = list(TSV_COLUMNS)
    assert columns[: len(expected_22)] == expected_22, (
        f"v0.2.5 22-col contract broken on {dataset.name}: got {columns[:len(expected_22)]}"
    )

    # 6 BidsGuess columns appended.
    for col in BIDS_GUESS_COLUMNS:
        assert col in columns, f"missing BidsGuess column {col!r} on {dataset.name}"

    # Every populated proposed_basename must validate against the BIDS
    # schema, except:
    #   - derivatives rows (``proposed_datatype`` starts with
    #     ``"derivatives/"``); these live outside the raw BIDS validation
    #     surface by design.
    #   - rows whose ``proposed_issues`` already records the schema verdict
    #     (e.g. bold without task gets a placeholder + an issue note).
    populated = written[written["proposed_basename"].astype(str) != ""]
    for _, row in populated.iterrows():
        if str(row.get("proposed_datatype") or "").startswith("derivatives/"):
            continue
        verdicts = bids_schema.validate_basename(
            row["proposed_basename"], row["proposed_datatype"]
        )
        errors = [v for v in verdicts if v.severity is bids_schema.Severity.ERROR]
        if errors and not str(row.get("proposed_issues") or ""):
            pytest.fail(
                f"schema rejected proposed basename {row['proposed_basename']!r} "
                f"(datatype={row['proposed_datatype']!r}) on {dataset.name}: {errors}"
            )

    # If no DICOMs were found at all, the rest of the assertions don't apply
    # (e.g. an empty placeholder folder); we only require the TSV to be
    # written and well-formed.
    if df.empty:
        return

    # Subject IDs must follow the v0.2.5 'sub-NNN' contract.
    assert (written["BIDS_name"].str.startswith("sub-")).all(), (
        f"non-conforming BIDS_name on {dataset.name}"
    )


def test_ppmi_longitudinal_sessions_split():
    """PPMI patients have multiple visits identified by ``StudyInstanceUID +
    StudyDate``. The scanner must merge them into one ``sub-NNN`` (same
    ``PatientID + PatientName``) and split into ``ses-1`` / ``ses-2`` /…
    in chronological order.
    """
    if not REAL_MRI_ROOT.exists():
        pytest.skip("real MRI dataset root missing")
    ppmi = REAL_MRI_ROOT / "PPMI"
    if not ppmi.exists():
        pytest.skip("PPMI dataset missing")

    out = REAL_MRI_ROOT.parent / "_pytest_ppmi_longitudinal.tsv"
    try:
        run_scan(ppmi, out, n_jobs=4)
        df = pd.read_csv(out, sep="\t", keep_default_na=False, dtype=str)
    finally:
        try:
            out.unlink()
        except FileNotFoundError:
            pass

    # PPMI's two known patients should produce two subjects.
    subjects = df["BIDS_name"].unique()
    assert len(subjects) == 2, f"expected 2 PPMI subjects, got {sorted(subjects)}"

    # Each subject should have at least 2 distinct sessions (ses-1, ses-2).
    for sub in subjects:
        sub_df = df[df["BIDS_name"] == sub]
        sessions = {s for s in sub_df["session"] if s}
        assert len(sessions) >= 2, (
            f"{sub} has only {sessions} sessions — longitudinal split failed"
        )
        # And the session labels must follow ses-N format.
        for s in sessions:
            assert s.startswith("ses-"), f"unexpected session label: {s!r}"

    # The proposed_basename for at least one row must include the session token.
    populated = df[df["proposed_basename"].astype(str) != ""]
    assert any("_ses-" in name for name in populated["proposed_basename"]), (
        "no proposed_basename includes a session token"
    )


def test_rep_column_chronological_within_groups():
    """Rep column must be the chronological position within each
    ``(BIDS_name, session, sequence, image_type)`` group of size > 1.

    PPMI ses-1 has 5 ``2D GRE-MT`` acquisitions of the same image_type
    in time order — they must be numbered ``1, 2, 3, 4, 5``.
    """
    if not REAL_MRI_ROOT.exists():
        pytest.skip("real MRI dataset root missing")
    ppmi = REAL_MRI_ROOT / "PPMI"
    if not ppmi.exists():
        pytest.skip("PPMI dataset missing")
    out = REAL_MRI_ROOT.parent / "_pytest_rep_chronological.tsv"
    try:
        run_scan(ppmi, out, n_jobs=4)
        df = pd.read_csv(out, sep="\t", keep_default_na=False, dtype=str)
    finally:
        try:
            out.unlink()
        except FileNotFoundError:
            pass

    grp = df[
        (df["BIDS_name"] == "sub-001")
        & (df["session"] == "ses-1")
        & (df["sequence"] == "2D GRE-MT")
        & (df["image_type"] == "M")
    ].sort_values(["acq_time", "series_uid"])
    assert len(grp) >= 2, "expected the PPMI multi-acquisition GRE-MT cluster"
    reps = list(grp["rep"])
    assert reps == [str(i + 1) for i in range(len(grp))], (
        f"rep column not chronological: {reps}"
    )


def test_neuroimging_old_dwi_scanner_derivatives_classified_correctly():
    """neuroimging_old has explicit ``..._FA``, ``..._TRACEW``, ``..._ColFA``,
    and ``..._TENSOR`` series. The classifier must emit BIDS scanner-
    derivative suffixes (FA/colFA/trace) for the first three and route
    TENSOR to ``derivatives/``.

    The same dataset has a 75-file ``acq-1b0`` DWI sitting next to an 8775-
    file ``acq-15`` peer; the cross-row B0 detector must reroute it to
    ``fmap/epi``.
    """
    if not REAL_MRI_ROOT.exists():
        pytest.skip("real MRI dataset root missing")
    src = REAL_MRI_ROOT / "neuroimging_old"
    if not src.exists():
        pytest.skip("neuroimging_old missing")
    out = REAL_MRI_ROOT.parent / "_pytest_neuroimging_old.tsv"
    try:
        run_scan(src, out, n_jobs=4)
        df = pd.read_csv(out, sep="\t", keep_default_na=False, dtype=str)
    finally:
        try:
            out.unlink()
        except FileNotFoundError:
            pass

    # FA / colFA / trace must appear with the right suffix.
    expected_pairs = {
        "ses-pre_dir-ap_dwi_FA": ("dwi", "FA"),
        "ses-pre_dir-ap_dwi_TRACEW": ("dwi", "trace"),
        "ses-pre_dir-ap_dwi_ColFA": ("dwi", "colFA"),
        "ses-pre_dir-ap_dwi_TENSOR": ("derivatives", "TENSOR"),
    }
    for sequence, (want_dt, want_suf) in expected_pairs.items():
        match = df[df["sequence"] == sequence]
        if match.empty:
            continue
        row = match.iloc[0]
        assert row["bids_guess_datatype"] == want_dt, (
            f"{sequence}: datatype={row['bids_guess_datatype']!r} "
            f"(expected {want_dt!r})"
        )
        assert row["bids_guess_suffix"] == want_suf, (
            f"{sequence}: suffix={row['bids_guess_suffix']!r} "
            f"(expected {want_suf!r})"
        )

    # The ``acq-15_acq-1b0_dir-ap_dwi`` row must be rerouted to fmap/epi.
    b0_row = df[df["sequence"] == "acq-15_acq-1b0_dir-ap_dwi"]
    if not b0_row.empty:
        r = b0_row.iloc[0]
        assert r["bids_guess_datatype"] == "fmap"
        assert r["bids_guess_suffix"] == "epi"
        assert "rerouted to fmap/epi" in r["proposed_issues"]


def test_repetition_type_column_present():
    """Every dataset's TSV must have the new ``repetition_type`` column."""
    if not REAL_MRI_ROOT.exists():
        pytest.skip("real MRI dataset root missing")
    sample = REAL_MRI_ROOT / "neuroimaging_unit_new"
    if not sample.exists():
        pytest.skip("sample dataset missing")
    out = REAL_MRI_ROOT.parent / "_pytest_reptype.tsv"
    try:
        run_scan(sample, out, n_jobs=4)
        df = pd.read_csv(out, sep="\t", keep_default_na=False, dtype=str)
    finally:
        try:
            out.unlink()
        except FileNotFoundError:
            pass
    assert "repetition_type" in df.columns
    valid = {"", "isolated", "planned", "suspected_abort"}
    bad = set(df["repetition_type"].unique()) - valid
    assert not bad, f"unexpected repetition_type values: {bad}"
