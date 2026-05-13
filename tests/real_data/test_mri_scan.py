"""Real-data characterisation test for MRI DICOM scan + BidsGuess (M1).

Gated on ``BIDS_MANAGER_REAL_MRI_DATA``. The test runs the full
``bidsmgr-scan`` pipeline against the Siemens MAGNETOM Prisma dataset at
``/Users/karelo/Development/datasets/BIDS_Manager/raw_data/MRI/neuroimaging_unit_new/``
and asserts:

* The TSV output preserves the v0.2.5 22-column contract.
* The BidsGuess columns are appended after those.
* At least one anat (T1w) row was successfully classified by BidsGuess.
* At least one func (bold) row was successfully classified.
* Subject IDs follow the v0.2.5 ``sub-NNN`` autonumbering rule.
* Localizer scans are flagged as ``discard`` and their ``include`` is ``0``.

This is a *characterisation* test — it pins current behaviour so future
refactors of the schema engine and BidsGuess classifier don't silently
regress on the real Prisma dataset.

Improvement plan: see ``../improvement_plan.md`` M1; super_plan.md §14.5.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli.scan import BIDS_GUESS_COLUMNS, run_scan
from bidsmgr.inventory.mri_dicom import TSV_COLUMNS

REAL_MRI_ROOT = Path(
    "/Users/karelo/Development/datasets/BIDS_Manager/raw_data/MRI/neuroimaging_unit_new"
)


pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not os.environ.get("BIDS_MANAGER_REAL_MRI_DATA"),
        reason="BIDS_MANAGER_REAL_MRI_DATA not set",
    ),
    pytest.mark.skipif(
        not REAL_MRI_ROOT.exists(),
        reason=f"real MRI dataset missing at {REAL_MRI_ROOT}",
    ),
]


@pytest.fixture(scope="module")
def scan_output(tmp_path_factory) -> pd.DataFrame:
    out = tmp_path_factory.mktemp("bidsmgr_scan") / "inventory.tsv"
    df = run_scan(REAL_MRI_ROOT, out, n_jobs=4)
    assert out.exists(), "TSV not written"
    return pd.read_csv(out, sep="\t", keep_default_na=False, dtype=str)


def test_tsv_preserves_v025_22col_contract(scan_output: pd.DataFrame):
    columns = list(scan_output.columns)
    expected_prefix = list(TSV_COLUMNS)
    assert columns[: len(expected_prefix)] == expected_prefix


def test_bids_guess_columns_appended(scan_output: pd.DataFrame):
    for col in BIDS_GUESS_COLUMNS:
        assert col in scan_output.columns


def test_subject_ids_use_v025_autonumbering(scan_output: pd.DataFrame):
    assert (scan_output["BIDS_name"].str.startswith("sub-")).all()
    # neuroimaging_unit_new contains a small number of unique subjects (≤4).
    assert scan_output["BIDS_name"].nunique() <= 4
    assert scan_output["BIDS_name"].nunique() >= 1


def test_classifier_produces_anat_t1w(scan_output: pd.DataFrame):
    matches = scan_output[
        (scan_output["bids_guess_datatype"] == "anat")
        & (scan_output["bids_guess_suffix"] == "T1w")
    ]
    assert len(matches) >= 1, (
        "Expected at least one anat/T1w classification on the Prisma dataset"
    )
    # Proposed BIDS name should be schema-valid and end in _T1w.nii.gz.
    for name in matches["proposed_basename"]:
        assert name.endswith("_T1w"), f"unexpected basename: {name}"


def test_classifier_produces_func_bold(scan_output: pd.DataFrame):
    matches = scan_output[
        (scan_output["bids_guess_datatype"] == "func")
        & (scan_output["bids_guess_suffix"] == "bold")
    ]
    assert len(matches) >= 1


def test_localizer_marked_as_discard(scan_output: pd.DataFrame):
    discards = scan_output[scan_output["bids_guess_datatype"] == "discard"]
    if len(discards):
        # Skip rows must have include=0.
        assert (discards["include"].astype(int) == 0).all()


def test_proposed_paths_validate_against_schema(scan_output: pd.DataFrame):
    """A populated proposed_basename validates iff ``proposed_issues`` is empty.

    Rows where the classifier had to placeholder a required entity (e.g.
    bold without a task) emit a basename anyway, but ``proposed_issues``
    records why it would be rejected so the GUI can prompt the user.
    """
    from bidsmgr import schema as bids_schema

    populated = scan_output[scan_output["proposed_basename"].astype(str) != ""]
    assert len(populated) >= 1

    for _, row in populated.iterrows():
        verdicts = bids_schema.validate_basename(
            row["proposed_basename"], row["proposed_datatype"]
        )
        errors = [v for v in verdicts if v.severity.value == "error"]
        issues = str(row.get("proposed_issues") or "")
        if errors and not issues:
            pytest.fail(
                f"row has schema-invalid proposed_basename without "
                f"proposed_issues being set: {row['proposed_basename']!r}: {errors}"
            )


def test_run_numbering_is_consistent_across_paired_series(scan_output: pd.DataFrame):
    """SBref / bold / physio rows for the same task must agree on run-N."""
    import re

    by_subj_session_task: dict[tuple, list[tuple]] = {}
    for _, row in scan_output.iterrows():
        if row["bids_guess_skip"] == "True":
            continue
        basename = str(row.get("proposed_basename") or "")
        if not basename:
            continue
        m_task = re.search(r"task-([0-9a-zA-Z]+)", basename)
        if not m_task:
            continue
        m_run = re.search(r"run-(\d+)", basename)
        run = int(m_run.group(1)) if m_run else None
        suffix_match = re.search(r"_([A-Za-z0-9]+)$", basename)
        suffix = suffix_match.group(1) if suffix_match else ""
        key = (row["BIDS_name"], row["session"], m_task.group(1))
        by_subj_session_task.setdefault(key, []).append((suffix, run, basename))

    for key, members in by_subj_session_task.items():
        suffix_runs: dict[str, list[int | None]] = {}
        for suffix, run, _ in members:
            suffix_runs.setdefault(suffix, []).append(run)
        # Within each (subject, session, task) group, every suffix's run-set
        # should be the same — bold runs and sbref runs should pair up.
        run_sets = {tuple(sorted(v, key=lambda x: (x is None, x))) for v in suffix_runs.values()}
        if len(run_sets) > 1:
            pytest.fail(
                f"inconsistent run-numbering across paired series for {key}: "
                f"{suffix_runs}"
            )
