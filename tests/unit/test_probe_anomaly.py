"""Unit tests for ``cli.scan._probe_anomaly``.

The probe-convert pass records actual dcm2niix outputs per
``SeriesInstanceUID``. ``_probe_anomaly`` compares the produced NIfTI
count against the (datatype, suffix) expectation and returns a
human-readable note when something looks wrong (e.g. ``func/bold`` got
2 NIfTIs from 1 input DICOM series — the operator-aborted-volume case
the user asked about).
"""

from __future__ import annotations

from bidsmgr.cli.scan import _probe_anomaly


def test_bold_one_nifti_one_uid_no_anomaly():
    assert _probe_anomaly("func", "bold", n_nifti=1, n_uids=1) is None


def test_bold_two_nifti_one_uid_flags_split():
    """The user's headline case: a bold task that the technician aborted
    produced an extra volume that dcm2niix split into a separate NIfTI."""
    msg = _probe_anomaly("func", "bold", n_nifti=2, n_uids=1)
    assert msg is not None
    assert "split conversion" in msg or "2 NIfTI" in msg


def test_bold_zero_nifti_one_uid_flags_missing():
    msg = _probe_anomaly("func", "bold", n_nifti=0, n_uids=1)
    assert msg is not None
    assert "missing" in msg or "expected 1" in msg


def test_dwi_one_nifti_one_uid_no_anomaly():
    """DWI produces 1 NIfTI + bval + bvec; only the NIfTI count matters."""
    assert _probe_anomaly("dwi", "dwi", n_nifti=1, n_uids=1) is None


def test_fmap_phasediff_three_nifti_no_anomaly():
    """Gradient fmap produces mag1 + mag2 + phasediff from one input
    series; that's a documented variable case (expectation=None)."""
    assert _probe_anomaly("fmap", "phasediff", n_nifti=3, n_uids=1) is None
    # Even an unusual count (e.g. 2) doesn't flag — fmap expectations vary.
    assert _probe_anomaly("fmap", "phasediff", n_nifti=2, n_uids=1) is None


def test_fmap_epi_one_nifti_no_anomaly():
    """PEpolar fmaps produce a single NIfTI."""
    assert _probe_anomaly("fmap", "epi", n_nifti=1, n_uids=1) is None
    msg = _probe_anomaly("fmap", "epi", n_nifti=2, n_uids=1)
    assert msg is not None  # split → flag


def test_t1w_one_nifti_no_anomaly():
    assert _probe_anomaly("anat", "T1w", n_nifti=1, n_uids=1) is None


def test_anat_megre_variable_no_anomaly():
    """MEGRE is multi-echo; the NIfTI count varies per protocol — don't flag."""
    assert _probe_anomaly("anat", "MEGRE", n_nifti=1, n_uids=1) is None
    assert _probe_anomaly("anat", "MEGRE", n_nifti=4, n_uids=1) is None


def test_unknown_datatype_no_anomaly():
    """Datatypes / suffixes we don't have an expectation for are silent."""
    assert _probe_anomaly("foo", "bar", n_nifti=99, n_uids=1) is None


def test_collapsed_fmap_row_with_three_uids():
    """The fmap collapse step joins multiple UIDs (mag1 / mag2 / phasediff)
    into one DataFrame row. The probe's NIfTI count is the SUM across UIDs;
    expected is also scaled by ``n_uids``."""
    # Three UIDs, each producing 1 NIfTI through fmap.epi → expected 3, got 3.
    assert _probe_anomaly("fmap", "epi", n_nifti=3, n_uids=3) is None
