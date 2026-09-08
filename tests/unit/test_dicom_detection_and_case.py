"""Two detection fixes that PET support depends on, with MRI regressions.

Both bugs predate PET and both also affect MRI, so the regressions here matter
as much as the new behaviour:

1. ``classifier.dcm2niix_bidsguess.canonicalise`` resolves dcm2niix's casing to
   the schema's. dcm2niix emits ``["PET", "PET"]`` while the schema spells both
   ``pet``, so a case-sensitive check silently rejected every PET series. The
   resolve must be case-INSENSITIVE, not a lowercase: the schema is full of
   mixed-case suffixes (``T1w``, ``FLAIR``) that ``.lower()`` would destroy.

2. ``inventory.mri_dicom.is_dicom_file`` used to reject any filename containing
   a dot without opening it. Philips PET writes the SOP instance UID as the
   filename and GE Signa writes ``*.img``; both are real DICOM and both were
   skipped, whatever the modality.
"""

from __future__ import annotations

import pytest

from bidsmgr import schema
from bidsmgr.classifier.dcm2niix_bidsguess import (
    _validate_classification,
    canonicalise,
    parse_bids_guess,
)
from bidsmgr.inventory.mri_dicom import is_dicom_file


# ---------------------------------------------------------------------------
# 1. dcm2niix casing
# ---------------------------------------------------------------------------


def test_pet_bidsguess_is_accepted() -> None:
    """The bug: PET series were silently dropped by the schema check."""
    datatype, entities, suffix = parse_bids_guess(["PET", "PET"])
    datatype, suffix = canonicalise(datatype, suffix)

    assert (datatype, suffix) == ("pet", "pet")
    assert _validate_classification(datatype, suffix, entities)


@pytest.mark.parametrize(
    "guess, expected",
    [
        (["anat", "_T1w"], ("anat", "T1w")),
        (["anat", "_T2w"], ("anat", "T2w")),
        (["anat", "_FLAIR"], ("anat", "FLAIR")),
        (["func", "_task-rest_bold"], ("func", "bold")),
        (["dwi", "_dwi"], ("dwi", "dwi")),
        (["fmap", "_epi"], ("fmap", "epi")),
        (["PET", "PET"], ("pet", "pet")),
    ],
)
def test_canonicalise_preserves_schema_spelling(guess, expected) -> None:
    """REGRESSION: a blanket ``.lower()`` would turn ``T1w`` into ``t1w``."""
    datatype, entities, suffix = parse_bids_guess(guess)
    assert canonicalise(datatype, suffix) == expected
    assert _validate_classification(*canonicalise(datatype, suffix), entities)


def test_canonicalise_leaves_discard_alone() -> None:
    assert canonicalise("discard", "discard") == ("discard", "discard")


def test_canonicalise_passes_through_unknown_tokens() -> None:
    """No case-insensitive match means no change, so the schema still rejects."""
    assert canonicalise("notadatatype", "notasuffix") == ("notadatatype", "notasuffix")
    assert not _validate_classification("notadatatype", "notasuffix", {})


def test_pet_entities_are_schema_valid() -> None:
    """PET carries its own entities; make sure the schema agrees."""
    allowed = set(schema.allowed_entities("pet", "pet"))
    assert {"tracer", "reconstruction"} <= allowed


# ---------------------------------------------------------------------------
# 2. DICOM file detection
# ---------------------------------------------------------------------------


def _write_dicomish(path, *, marker: bool = True) -> str:
    """A 132-byte stub: 128 bytes of preamble then the ``DICM`` marker."""
    path.write_bytes(b"\0" * 128 + (b"DICM" if marker else b"JUNK") + b"\0" * 32)
    return str(path)


def test_dotted_uid_filename_is_detected(tmp_path) -> None:
    """The bug: Philips PET writes the SOP instance UID as the filename."""
    f = _write_dicomish(tmp_path / "1.3.46.670589.28.2.15.30.26461.41445.3.3400.628.1500291755")
    assert is_dicom_file(f)


def test_dotted_img_filename_is_detected(tmp_path) -> None:
    """GE Signa PET/MR writes ``i439953.PTDC.89.img``, real DICOM."""
    assert is_dicom_file(_write_dicomish(tmp_path / "i439953.PTDC.89.img"))


def test_known_extensions_still_shortcut(tmp_path) -> None:
    """``.dcm``/``.ima`` are accepted without reading the file at all."""
    for name in ("scan.dcm", "SCAN.DCM", "scan.IMA"):
        p = tmp_path / name
        p.write_bytes(b"not really dicom")
        assert is_dicom_file(str(p))


def test_extensionless_marker_still_works(tmp_path) -> None:
    """REGRESSION: Canon and GE Discovery write extensionless names."""
    assert is_dicom_file(_write_dicomish(tmp_path / "O0001025"))


def test_file_without_marker_is_rejected(tmp_path) -> None:
    assert not is_dicom_file(_write_dicomish(tmp_path / "O0001025", marker=False))


def test_short_file_is_rejected(tmp_path) -> None:
    p = tmp_path / "tiny"
    p.write_bytes(b"abc")
    assert not is_dicom_file(str(p))


@pytest.mark.parametrize(
    "name",
    [
        "report.pdf",           # the 4D/BTi collision that bit the EEG scanner
        "notes.txt",
        "archive.zip",
        "screenshot.png",
        "recording.fif",        # belongs to the EEG/MEG scanner
        "raw.edf",
        "series.nii",
        "config.json",
    ],
)
def test_known_non_dicom_extensions_are_rejected_without_reading(tmp_path, name) -> None:
    """REGRESSION: these must stay out even if their bytes look like DICOM."""
    assert not is_dicom_file(_write_dicomish(tmp_path / name))


def test_missing_file_is_rejected(tmp_path) -> None:
    assert not is_dicom_file(str(tmp_path / "does_not_exist"))
