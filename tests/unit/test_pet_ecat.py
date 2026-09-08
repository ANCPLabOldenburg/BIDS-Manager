"""ECAT PET: signature detection, header probing, backend dispatch.

ECAT is PET's non-DICOM native format (Siemens HRRT and older CTI scanners).
dcm2niix cannot read it; nibabel can, and nibabel is already a dependency, so
the format costs nothing extra to support.

The real-data half of this file is gated on ``BIDS_MANAGER_REAL_PET_DATA=1``
like the other real-data suites.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bidsmgr.converter.backends.ecat_direct import EcatDirect
from bidsmgr.converter.types import ConvertTask
from bidsmgr.inventory.pet_ecat import (
    ECAT_MAGIC,
    _decode,
    _number,
    find_ecat_files,
    is_ecat_file,
)

REAL_DATA = os.environ.get("BIDS_MANAGER_REAL_PET_DATA") == "1"
# The three ECAT phantoms, in the raw data where they persist. This used to
# point at a scratch directory under bids_manager_outputs, which is conversion
# OUTPUT and gets cleared, so these tests silently found nothing the moment the
# folder was tidied away.
PHANTOMS = Path(
    "/Users/karelo/Development/datasets/BIDS_Manager/raw_data/PET_DICOMS"
    "/PN000001/OpenNeuroPET-Phantoms/sourcedata"
)


# ---------------------------------------------------------------------------
# Signature detection
# ---------------------------------------------------------------------------


def _write_ecat(path: Path, version: bytes = b"3") -> Path:
    """A stub carrying the ECAT7 signature. Enough for detection tests."""
    path.write_bytes(ECAT_MAGIC + version + b"v" + b"\0" * 500)
    return path


def test_ecat_signature_is_detected(tmp_path) -> None:
    assert is_ecat_file(_write_ecat(tmp_path / "scan.v"))


@pytest.mark.parametrize("version", [b"0", b"2", b"3"])
def test_every_ecat7_subversion_is_detected(tmp_path, version) -> None:
    """MATRIX70, 72 and 73 all appear across the phantom scanners."""
    assert is_ecat_file(_write_ecat(tmp_path / f"scan{version.decode()}.v", version))


def test_extension_alone_is_not_enough(tmp_path) -> None:
    """``.v`` is used by Verilog and others, so the magic has to decide."""
    p = tmp_path / "module.v"
    p.write_text("module counter(input clk); endmodule")
    assert not is_ecat_file(p)


def test_unrelated_extension_is_skipped_without_reading(tmp_path) -> None:
    assert not is_ecat_file(_write_ecat(tmp_path / "scan.txt"))


def test_missing_file_is_not_ecat(tmp_path) -> None:
    assert not is_ecat_file(tmp_path / "nope.v")


def test_find_ecat_files_walks_recursively(tmp_path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    _write_ecat(tmp_path / "a" / "one.v")
    _write_ecat(tmp_path / "a" / "b" / "two.v")
    (tmp_path / "a" / "notes.txt").write_text("hi")

    found = find_ecat_files(tmp_path)
    assert {p.name for p in found} == {"one.v", "two.v"}
    # Sorted by full path, so the order is stable across runs and platforms.
    assert found == sorted(found)


# ---------------------------------------------------------------------------
# Header value unwrapping
# ---------------------------------------------------------------------------


def test_decode_unwraps_numpy_bytes() -> None:
    """nibabel returns 0-d numpy scalars; a bare str() leaves the numpy repr."""
    import numpy as np

    assert _decode(np.bytes_(b"F-18\0\0\0")) == "F-18"
    assert _decode(b"Rigshospitalet\0") == "Rigshospitalet"
    assert _decode(None) == ""


def test_number_unwraps_numpy_scalars() -> None:
    import numpy as np

    assert _number(np.float32(2.5)) == pytest.approx(2.5)
    assert _number(None) is None
    assert _number(b"not a number") is None


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def _task(tmp_path: Path, source: Path, datatype: str = "pet") -> ConvertTask:
    return ConvertTask(
        row_id="r1", series_uid="", source_files=(source,),
        dataset="ds", bids_root=tmp_path / "bids", subject="001",
        datatype=datatype, suffix="pet", basename="sub-001_pet",
    )


def test_backend_claims_a_pet_row_with_an_ecat_source(tmp_path) -> None:
    src = _write_ecat(tmp_path / "scan.v")
    assert EcatDirect().can_handle(_task(tmp_path, src))


def test_backend_declines_a_pet_row_from_dicom(tmp_path) -> None:
    """DICOM PET must keep going to dcm2niix, not to this backend."""
    src = tmp_path / "scan.dcm"
    src.write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 64)
    assert not EcatDirect().can_handle(_task(tmp_path, src))


def test_backend_declines_non_pet_datatypes(tmp_path) -> None:
    src = _write_ecat(tmp_path / "scan.v")
    assert not EcatDirect().can_handle(_task(tmp_path, src, datatype="anat"))


def test_registry_puts_ecat_ahead_of_the_dcm2niix_fallback() -> None:
    from bidsmgr.converter.registry import default_backends

    names = [b.name for b in default_backends()]
    assert names.index("ecat_direct") < names.index("dcm2niix_direct")


def test_unreadable_ecat_reports_an_error_rather_than_raising(tmp_path) -> None:
    """A truncated file must fail the task, never kill the run."""
    src = _write_ecat(tmp_path / "truncated.v")
    result = EcatDirect().convert(_task(tmp_path, src), tmp_path / "staging")
    assert not result.success
    assert result.error


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_DATA, reason="needs BIDS_MANAGER_REAL_PET_DATA=1")
def test_real_phantoms_probe_cleanly() -> None:
    from bidsmgr.inventory.pet_ecat import probe_ecat

    files = find_ecat_files(PHANTOMS)
    assert len(files) == 3, "expected the three ECAT phantoms"
    for fp in files:
        probe = probe_ecat(fp)
        assert probe is not None
        assert probe.n_frames >= 1
        assert len(probe.shape) == 4
        assert probe.frame_durations
        # Durations are converted from the header's milliseconds to seconds.
        assert all(d > 0 for d in probe.frame_durations)


def test_unrecorded_orientation_is_corrected(tmp_path) -> None:
    """nibabel flips ECAT data by patient_orientation but builds its affine
    without consulting it, so an unrecorded orientation (code 8) leaves data
    and affine disagreeing. That image reads upside down and mirrored."""
    import numpy as np

    from bidsmgr.converter.backends.ecat_direct import _orient_to_affine

    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)

    class _Img:
        def __init__(self, code):
            self.header = {"patient_orientation": np.array(code)}

    corrected = _orient_to_affine(_Img(8), data)
    assert np.array_equal(corrected, np.flip(data, axis=(0, 1, 2)))


@pytest.mark.parametrize("code", [0, 1, 2, 3, 4, 5, 6, 7])
def test_a_recorded_orientation_is_left_to_nibabel(tmp_path, code) -> None:
    """REGRESSION: nibabel already flipped these; flipping again would break
    the files that are currently correct."""
    import numpy as np

    from bidsmgr.converter.backends.ecat_direct import _orient_to_affine

    data = np.arange(24, dtype=float).reshape(2, 3, 4)

    class _Img:
        def __init__(self, c):
            self.header = {"patient_orientation": np.array(c)}

    assert np.array_equal(_orient_to_affine(_Img(code), data), data)


def test_a_missing_orientation_field_is_survived() -> None:
    import numpy as np

    from bidsmgr.converter.backends.ecat_direct import _orient_to_affine

    data = np.zeros((2, 2, 2))

    class _Img:
        header: dict = {}

    assert np.array_equal(_orient_to_affine(_Img(), data), data)


@pytest.mark.skipif(not REAL_DATA, reason="needs BIDS_MANAGER_REAL_PET_DATA=1")
def test_a_single_frame_scan_is_written_as_3d(tmp_path) -> None:
    """A static scan must not look dynamic. nibabel reports ECAT as
    (x, y, z, frames) whatever the count, and a trailing length-1 axis makes
    anything keying on ndim treat a static image as a time series."""
    import nibabel

    src = find_ecat_files(PHANTOMS)[0]
    result = EcatDirect().convert(_task(tmp_path, src), tmp_path / "staging")
    assert result.success, result.error
    nii = next(p for p in result.staged_files if p.name.endswith(".nii.gz"))
    assert nibabel.load(str(nii)).ndim == 3


@pytest.mark.skipif(not REAL_DATA, reason="needs BIDS_MANAGER_REAL_PET_DATA=1")
def test_real_phantom_converts_with_matching_voxels(tmp_path) -> None:
    """The written NIfTI must carry the ECAT's scaled data, not raw counts."""
    import nibabel
    import numpy as np

    src = find_ecat_files(PHANTOMS)[0]
    result = EcatDirect().convert(_task(tmp_path, src), tmp_path / "staging")
    assert result.success, result.error

    nii = next(p for p in result.staged_files if p.name.endswith(".nii.gz"))
    written = nibabel.load(str(nii)).get_fdata()
    # Squeeze the frame axis and undo the orientation correction to recover
    # exactly what nibabel read, so this checks the data path rather than
    # restating the fix.
    from bidsmgr.converter.backends.ecat_direct import _orient_to_affine

    ecat = nibabel.ecat.load(str(src))
    expected = np.squeeze(_orient_to_affine(ecat, ecat.get_fdata()))
    assert written.shape == expected.shape
    assert np.allclose(written, expected, rtol=1e-5, atol=1e-8)
