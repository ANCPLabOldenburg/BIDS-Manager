"""What ``fixups/mrs_header`` does to a NIfTI-MRS file and its sidecar.

Two jobs: take the participant's identifiers out of the header extension,
where no sidecar pass can reach them, and stop a field whose zero means
"not done" from failing validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.fixups.mrs_header import (
    MRS_EXTENSION_CODE,
    clean_mrs_file,
    read_mrs_header,
)

nib = pytest.importorskip("nibabel")


def _write(tmp_path: Path, header: dict, sidecar: dict | None = None) -> Path:
    """A minimal genuine NIfTI-MRS file: complex FID, header in code 44."""
    folder = tmp_path / "sub-001" / "mrs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "sub-001_svs.nii.gz"
    data = (np.ones((1, 1, 1, 64)) + 1j * np.ones((1, 1, 1, 64))).astype(np.complex64)
    img = nib.Nifti2Image(data, np.eye(4))
    img.header.extensions.append(
        nib.nifti1.Nifti1Extension(MRS_EXTENSION_CODE, json.dumps(header).encode())
    )
    nib.save(img, str(path))
    if sidecar is not None:
        (folder / "sub-001_svs.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def _sidecar(path: Path) -> dict:
    return json.loads(path.with_name("sub-001_svs.json").read_text(encoding="utf-8"))


BASE = {"SpectrometerFrequency": [123.26], "ResonantNucleus": ["1H"]}


class TestAZeroInversionTime:
    """dcm2niix writes ``InversionTime: 0`` for single-voxel spectroscopy,
    which has no inversion pulse, and BIDS constrains the field to be above
    zero wherever it appears. Every spectroscopy dataset failed validation
    with an ERROR until this was removed."""

    def test_it_is_dropped_from_the_sidecar(self, tmp_path):
        path = _write(tmp_path, {**BASE, "InversionTime": 0},
                      sidecar={"InversionTime": 0, "EchoTime": 0.03})
        clean_mrs_file(path)
        side = _sidecar(path)
        assert "InversionTime" not in side
        assert side["EchoTime"] == 0.03, "its neighbours are untouched"

    def test_it_is_dropped_from_the_header_too(self, tmp_path):
        """Or the sidecar copy puts it straight back on the next pass."""
        path = _write(tmp_path, {**BASE, "InversionTime": 0}, sidecar={})
        clean_mrs_file(path)
        assert "InversionTime" not in read_mrs_header(path)
        clean_mrs_file(path)          # a second pass must not resurrect it
        assert "InversionTime" not in _sidecar(path)

    def test_a_float_zero_counts(self, tmp_path):
        path = _write(tmp_path, {**BASE, "InversionTime": 0.0},
                      sidecar={"InversionTime": 0.0})
        clean_mrs_file(path)
        assert "InversionTime" not in _sidecar(path)

    def test_a_real_inversion_time_is_kept(self, tmp_path):
        """An inversion-recovery acquisition has one, and it means something."""
        path = _write(tmp_path, {**BASE, "InversionTime": 1.8},
                      sidecar={"InversionTime": 1.8})
        clean_mrs_file(path)
        assert _sidecar(path)["InversionTime"] == 1.8
        assert read_mrs_header(path)["InversionTime"] == 1.8

    def test_the_spectrum_still_reads_after_the_rewrite(self, tmp_path):
        from bidsmgr.gui.widgets.mrs_spectrum import read_mrs

        path = _write(tmp_path, {**BASE, "InversionTime": 0, "DwellTime": 0.0005},
                      sidecar={})
        clean_mrs_file(path)
        data = read_mrs(path)
        assert data is not None
        assert data["nucleus"] == "1H"


class TestIdentifiers:
    def test_they_are_removed_from_the_header(self, tmp_path):
        path = _write(tmp_path, {**BASE, "PatientName": "Doe^Jane",
                                 "PatientBirthDate": "19700101"})
        removed = clean_mrs_file(path)
        header = read_mrs_header(path)
        assert "PatientName" not in header
        assert "PatientBirthDate" not in header
        assert set(removed) >= {"PatientName", "PatientBirthDate"}

    def test_a_clean_file_is_not_rewritten(self, tmp_path):
        """A conversion that rewrites every file it touches makes its own
        logs useless for telling which files it changed."""
        path = _write(tmp_path, dict(BASE), sidecar={})
        before = path.stat().st_mtime_ns
        assert clean_mrs_file(path) == []
        assert path.stat().st_mtime_ns == before

    def test_a_plain_nifti_is_left_alone(self, tmp_path):
        folder = tmp_path / "sub-001" / "anat"
        folder.mkdir(parents=True)
        path = folder / "sub-001_T1w.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), str(path))
        assert read_mrs_header(path) is None
        assert clean_mrs_file(path) == []
