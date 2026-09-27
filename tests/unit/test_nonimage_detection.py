"""Tests for non-image (no-pixel-data) DICOM detection.

A DICOM series whose files carry no Image Pixel module (Rows/Columns) is a
DERIVED non-image object (e.g. a Siemens diffusion ``TENSOR`` map or a
``PhoenixZIPReport``). dcm2niix cannot convert it to NIfTI, so the scanner
flags it: excluded from conversion (``include=0`` / ``bids_guess_skip``)
with a ``non-image series`` note in ``issues`` so it surfaces in
the inventory table highlighted rather than failing dcm2niix with ``rc=2``.

Covers the engine layer:
* ``mri_dicom._read_one`` stamps ``has_pixels`` per file.
* ``mri_dicom.scan_dicoms_long`` aggregates it into the internal
  ``_has_pixel_data`` column (OR across a series' files).
* ``cli/scan._flag_nonimage_rows`` excludes + annotates non-image rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from bidsmgr.cli.scan import (
    NONIMAGE_ISSUE_TOKEN,
    _flag_nonimage_rows,
    _is_nonimage_flag,
)
from bidsmgr.inventory.mri_dicom import _read_one, scan_dicoms_long


# ---------------------------------------------------------------------------
# Synthetic DICOM helper
# ---------------------------------------------------------------------------


def _write_dicom(
    path: Path,
    *,
    with_pixels: bool,
    series_desc: str,
    series_uid: str,
    patient_id: str = "P1",
    patient_name: str = "Doe^Jane",
) -> None:
    """Write a minimal DICOM with or without the Image Pixel module."""
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = MRImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.Modality = "MR"
    ds.SeriesDescription = series_desc
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    if with_pixels:
        ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
        ds.Rows = 4
        ds.Columns = 4
        ds.BitsAllocated = 16
        ds.PixelData = (b"\x00\x00") * 16
    else:
        # DERIVED non-image object: no Rows/Columns, no PixelData.
        ds.ImageType = ["DERIVED", "PRIMARY", "DIFFUSION", "TENSOR", "ND"]
    ds.save_as(str(path), enforce_file_format=True)


# ---------------------------------------------------------------------------
# _read_one pixel detection
# ---------------------------------------------------------------------------


def test_read_one_detects_pixels(tmp_path: Path) -> None:
    p = tmp_path / "img.dcm"
    _write_dicom(p, with_pixels=True, series_desc="t1_mprage", series_uid="1.1")
    res = _read_one(str(p), tmp_path)
    assert res is not None
    assert res["has_pixels"] is True


def test_read_one_detects_nonimage(tmp_path: Path) -> None:
    p = tmp_path / "tensor.dcm"
    _write_dicom(p, with_pixels=False, series_desc="dwi_TENSOR", series_uid="1.2")
    res = _read_one(str(p), tmp_path)
    assert res is not None
    assert res["has_pixels"] is False


# ---------------------------------------------------------------------------
# scan_dicoms_long aggregation -> internal _has_pixel_data column
# ---------------------------------------------------------------------------


def test_scan_aggregates_pixel_presence(tmp_path: Path) -> None:
    # One image series (2 files) + one non-image series (1 file), same subject.
    img_uid, ten_uid = "1.10", "1.20"
    _write_dicom(tmp_path / "a1.dcm", with_pixels=True, series_desc="t1_mprage", series_uid=img_uid)
    _write_dicom(tmp_path / "a2.dcm", with_pixels=True, series_desc="t1_mprage", series_uid=img_uid)
    _write_dicom(tmp_path / "t.dcm", with_pixels=False, series_desc="dwi_TENSOR", series_uid=ten_uid)

    df = scan_dicoms_long(tmp_path, n_jobs=1)
    assert "_has_pixel_data" in df.columns

    img = df[df["series_uid"] == img_uid].iloc[0]
    ten = df[df["series_uid"] == ten_uid].iloc[0]
    assert bool(img["_has_pixel_data"]) is True
    assert bool(ten["_has_pixel_data"]) is False


# ---------------------------------------------------------------------------
# _is_nonimage_flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "val,expected",
    [
        (False, True),          # explicit False -> non-image
        (True, False),          # explicit True -> image
        (float("nan"), False),  # fmap-collapsed / EEG rows -> image
        ("", False),
        ("False", True),        # stringified survives a round-trip
        ("True", False),
        (0, True),
        (1, False),
    ],
)
def test_is_nonimage_flag(val, expected: bool) -> None:
    assert _is_nonimage_flag(val) is expected


# ---------------------------------------------------------------------------
# _flag_nonimage_rows
# ---------------------------------------------------------------------------


def test_flag_nonimage_rows_excludes_and_annotates() -> None:
    df = pd.DataFrame([
        {"_has_pixel_data": True,  "include": 1, "bids_guess_skip": False, "issues": ""},
        {"_has_pixel_data": False, "include": 1, "bids_guess_skip": False, "issues": ""},
        {"_has_pixel_data": False, "include": 1, "bids_guess_skip": False, "issues": "trivial: x"},
    ])
    _flag_nonimage_rows(df)

    # Image row untouched.
    assert df.at[0, "include"] == 1
    assert not bool(df.at[0, "bids_guess_skip"])
    assert df.at[0, "issues"] == ""

    # Non-image row excluded + annotated.
    assert df.at[1, "include"] == 0
    assert bool(df.at[1, "bids_guess_skip"])
    assert NONIMAGE_ISSUE_TOKEN in df.at[1, "issues"]

    # Existing issue preserved after the prepended non-image note.
    assert NONIMAGE_ISSUE_TOKEN in df.at[2, "issues"]
    assert "trivial: x" in df.at[2, "issues"]


def test_flag_nonimage_rows_noop_without_column() -> None:
    df = pd.DataFrame([{"include": 1, "bids_guess_skip": False, "issues": ""}])
    _flag_nonimage_rows(df)  # no _has_pixel_data column -> no-op
    assert df.at[0, "include"] == 1


def test_physio_rows_are_exempt_from_nonimage_flag() -> None:
    """Siemens physio (_PhysioLog.dcm) has no pixel data either, but it IS
    convertible by the bidsphysio backend -> must NOT be flagged/excluded."""
    df = pd.DataFrame([
        # physio identified by suffix
        {"_has_pixel_data": False, "include": 1, "bids_guess_skip": False,
         "issues": "", "bids_guess_suffix": "physio", "modality": "physio",
         "bids_name": "sub-001_task-rest_physio"},
        # a genuine non-image object (TENSOR) for contrast
        {"_has_pixel_data": False, "include": 1, "bids_guess_skip": False,
         "issues": "", "bids_guess_suffix": "TENSOR", "modality": "dwi",
         "bids_name": "sub-001_desc-TENSOR_dwi"},
    ])
    _flag_nonimage_rows(df)

    # Physio row untouched.
    assert df.at[0, "include"] == 1
    assert not bool(df.at[0, "bids_guess_skip"])
    assert NONIMAGE_ISSUE_TOKEN not in df.at[0, "issues"]

    # TENSOR row still flagged.
    assert df.at[1, "include"] == 0
    assert bool(df.at[1, "bids_guess_skip"])
    assert NONIMAGE_ISSUE_TOKEN in df.at[1, "issues"]
