"""Tests for dcm2niix residual/secondary output detection + filtering.

dcm2niix splits a single input series into the real image PLUS derived
secondary volumes, naming the extras by gluing a collision letter onto the
``-f`` basename (``..._bold`` -> ``..._bolda``) or with an ``_Eq_`` / ``_ROI``
/ ``_i<instance>`` marker. Those are not real acquired images and have no
valid BIDS suffix, so by default the converter drops them. Legitimate
multi-output (fmap ``_e1``/``_e2``/``_ph``, complex parts, DWI bval/bvec)
must never be dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.converter.backends.dcm2niix_direct import (
    _is_residual_output,
    _name_phase_part,
    _partition_residuals,
)


BASENAME = "sub-001_task-dmaging_run-1_bold"


@pytest.mark.parametrize(
    "name,is_residual",
    [
        # Primary output + its sidecar -> keep.
        (f"{BASENAME}.nii.gz", False),
        (f"{BASENAME}.json", False),
        # The reproduced sina_data residual: glued collision letter -> drop.
        (f"{BASENAME}a.nii.gz", True),
        (f"{BASENAME}a.json", True),
        (f"{BASENAME}b.nii.gz", True),
        # dcm2niix reslice / scanner secondaries -> drop.
        (f"{BASENAME}_Eq_1.nii.gz", True),
        (f"{BASENAME}_Eq_2.json", True),
        (f"{BASENAME}_ROI1.nii.gz", True),
        (f"{BASENAME}_ROI.nii.gz", True),
        (f"{BASENAME}_i00001.nii.gz", True),
        # Legitimate multi-output siblings -> keep.
        (f"{BASENAME}_e1.nii.gz", False),
        (f"{BASENAME}_e2.nii.gz", False),
        (f"{BASENAME}_ph.nii.gz", False),
        (f"{BASENAME}_e2_ph.nii.gz", False),
        (f"{BASENAME}_real.nii.gz", False),
        (f"{BASENAME}_imaginary.nii.gz", False),
        (f"{BASENAME}.bval", False),
        (f"{BASENAME}.bvec", False),
    ],
)
def test_is_residual_output(name: str, is_residual: bool) -> None:
    assert _is_residual_output(Path(name), BASENAME) is is_residual


def test_partition_residuals_splits_keep_and_drop() -> None:
    staged = [
        Path(f"{BASENAME}.nii.gz"),
        Path(f"{BASENAME}.json"),
        Path(f"{BASENAME}a.nii.gz"),
        Path(f"{BASENAME}a.json"),
    ]
    keep, residual = _partition_residuals(staged, BASENAME)
    assert {p.name for p in keep} == {f"{BASENAME}.nii.gz", f"{BASENAME}.json"}
    assert {p.name for p in residual} == {f"{BASENAME}a.nii.gz", f"{BASENAME}a.json"}


def test_fmap_multi_output_never_dropped() -> None:
    fmap_base = "sub-001_acq-fm2_magnitude1"
    staged = [
        Path(f"{fmap_base}_e1.nii.gz"),
        Path(f"{fmap_base}_e1.json"),
        Path(f"{fmap_base}_e2.nii.gz"),
        Path(f"{fmap_base}_e2_ph.nii.gz"),
    ]
    keep, residual = _partition_residuals(staged, fmap_base)
    assert residual == []
    assert len(keep) == 4


def test_dwi_sidecars_never_dropped() -> None:
    dwi_base = "sub-001_dwi"
    staged = [
        Path(f"{dwi_base}.nii.gz"),
        Path(f"{dwi_base}.json"),
        Path(f"{dwi_base}.bval"),
        Path(f"{dwi_base}.bvec"),
    ]
    keep, residual = _partition_residuals(staged, dwi_base)
    assert residual == []
    assert len(keep) == 4


def test_a_part_phase_series_lands_at_its_planned_name(tmp_path) -> None:
    """dcm2niix appends ``_ph`` to a phase image even when the scan named the
    series ``part-phase``; the token goes, and nothing else is touched."""
    import json

    import nibabel as nib
    import numpy as np

    base = "sub-001_task-x_dir-PA_run-1_part-phase_bold"
    raw = np.array([-4096, 0, 4094], dtype="int16").reshape(1, 1, 3, 1)
    nib.save(nib.Nifti1Image(raw, np.eye(4)), str(tmp_path / f"{base}_ph.nii.gz"))
    (tmp_path / f"{base}_ph.json").write_text("{}")
    out = _name_phase_part(sorted(tmp_path.iterdir()), base)
    assert [p.name for p in out] == [f"{base}.json", f"{base}.nii.gz"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{base}.json", f"{base}.nii.gz"]
    # BIDS requires Units of a phase image; Siemens raw phase is not radians.
    assert json.loads((tmp_path / f"{base}.json").read_text())["Units"] == "arbitrary"
    rad = "sub-001_task-y_part-phase_bold"
    nib.save(nib.Nifti1Image(np.array([-3.14, 0.0, 3.14], dtype="float32").reshape(1, 1, 3),
                             np.eye(4)), str(tmp_path / f"{rad}.nii.gz"))
    (tmp_path / f"{rad}.json").write_text("{}")
    _name_phase_part([tmp_path / f"{rad}.json", tmp_path / f"{rad}.nii.gz"], rad)
    assert json.loads((tmp_path / f"{rad}.json").read_text())["Units"] == "rad"
    # A series mixing magnitude and phase keeps dcm2niix's names.
    mixed = "sub-001_task-x_run-1_sbref"
    for name in (f"{mixed}.nii.gz", f"{mixed}_ph.nii.gz"):
        (tmp_path / name).write_bytes(b"x")
    files = [tmp_path / f"{mixed}.nii.gz", tmp_path / f"{mixed}_ph.nii.gz"]
    assert _name_phase_part(files, mixed) == files
