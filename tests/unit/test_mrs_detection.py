"""MR spectroscopy, classified from the DICOM header rather than a conversion.

``mrs`` only ever arrived via dcm2niix's BidsGuess, and on Windows dcm2niix
does not survive spectroscopy: it dies with a stack overflow (``0xC00000FD``)
and an EMPTY stderr. Measured on two unrelated samples — one stored under the
standard MR Spectroscopy SOP class, one under Siemens' own CSA non-image class
— while ordinary images in the same studies converted fine.

The series then reached the inventory with no datatype, and a series with no
datatype is exactly the one ``_is_spectroscopy_row`` cannot rescue from the
non-image cull. In the Siemens sample it was worse than blank: the name
``task-xx_..._svs`` matched ``task-`` and the regex layer filed it as
``func``/``bold``, its reference scan as ``func``/``sbref``, and both were
dropped for carrying no pixel data.

A DICOM header does not crash, so the datatype no longer waits for a
converter. These tests cover both storage conventions, the three ways of
telling Siemens' shared CSA class apart, and the precedence rule.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from bidsmgr.cli.scan import _classify_mrs_rows, _mrs_may_overrule
from bidsmgr.inventory.mri_dicom import (
    MR_SPECTROSCOPY_SOP_CLASS,
    SIEMENS_CSA_NONIMAGE_SOP_CLASS,
    read_mrs_tags,
)

# The three CSA non-image objects one Siemens study ships, and what separates
# them. Measured, not assumed — see the module docstring of mri_dicom.
CSA_SPECTROSCOPY = ("SPEC NUM 4", ["ORIGINAL", "PRIMARY"])
CSA_PHYSIO = ("SPEC NUM 4", ["ORIGINAL", "PRIMARY", "RAWDATA", "PHYSIO"])
CSA_TENSOR = ("DTI NUM 4", ["DERIVED", "PRIMARY", "DIFFUSION", "TENSOR", "ND"])


def _dataset(
    *,
    sop_class: str = MRImageStorage,
    image_type=None,
    csa_type: str = "",
    rows=None,
    columns=None,
    localisation: str = "",
    nucleus: str = "",
) -> FileDataset:
    """A header carrying only what the detector is allowed to look at."""
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = MRImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("mem", {}, file_meta=fm, preamble=b"\0" * 128)
    ds.Modality = "MR"
    ds.SOPClassUID = sop_class
    if image_type is not None:
        ds.ImageType = list(image_type)
    if rows is not None:
        ds.Rows = rows
    if columns is not None:
        ds.Columns = columns
    if localisation:
        ds.VolumeLocalizationTechnique = localisation
    if nucleus:
        ds.ResonantNucleus = nucleus
    if csa_type:
        # Siemens' private CSA Image Type, (0029,1008).
        ds.add_new((0x0029, 0x1008), "CS", csa_type)
    return ds


# ---------------------------------------------------------------------------
# read_mrs_tags: what counts as spectroscopy
# ---------------------------------------------------------------------------


class TestDetection:
    def test_the_standard_sop_class_is_enough(self) -> None:
        """The authoritative marker: stored as spectroscopy, so it is."""
        tags = read_mrs_tags(_dataset(
            sop_class=MR_SPECTROSCOPY_SOP_CLASS,
            image_type=["ORIGINAL", "PRIMARY", "SPECTROSCOPY", "NONE"],
            rows=1, columns=1, localisation="PRESS", nucleus="1H",
        ))
        assert tags
        assert tags["localisation"] == "PRESS"
        assert tags["nucleus"] == "1H"

    def test_an_image_type_of_spectroscopy_is_enough(self) -> None:
        """A vendor that marks the frame but stores it elsewhere."""
        assert read_mrs_tags(_dataset(
            image_type=["ORIGINAL", "PRIMARY", "SPECTROSCOPY"],
        ))

    def test_siemens_csa_spectroscopy_is_recognised(self) -> None:
        """No Rows, no localisation, ImageType says only ORIGINAL/PRIMARY.
        The private CSA type is the only thing that identifies it."""
        csa_type, image_type = CSA_SPECTROSCOPY
        assert read_mrs_tags(_dataset(
            sop_class=SIEMENS_CSA_NONIMAGE_SOP_CLASS,
            image_type=image_type, csa_type=csa_type,
        ))

    def test_a_physio_log_is_not_spectroscopy(self) -> None:
        """Shares the SOP class AND the "SPEC" CSA type with spectroscopy, so
        the CSA type alone would swallow it. ImageType is what separates
        them."""
        csa_type, image_type = CSA_PHYSIO
        assert read_mrs_tags(_dataset(
            sop_class=SIEMENS_CSA_NONIMAGE_SOP_CLASS,
            image_type=image_type, csa_type=csa_type,
        )) == {}

    def test_a_tensor_map_is_not_spectroscopy(self) -> None:
        """Shares the SOP class; the CSA type is what separates it."""
        csa_type, image_type = CSA_TENSOR
        assert read_mrs_tags(_dataset(
            sop_class=SIEMENS_CSA_NONIMAGE_SOP_CLASS,
            image_type=image_type, csa_type=csa_type,
        )) == {}

    def test_the_csa_class_alone_proves_nothing(self) -> None:
        csa = _dataset(
            sop_class=SIEMENS_CSA_NONIMAGE_SOP_CLASS,
            image_type=["ORIGINAL", "PRIMARY"],
        )
        assert read_mrs_tags(csa) == {}

    def test_an_ordinary_image_is_not_spectroscopy(self) -> None:
        assert read_mrs_tags(_dataset(
            image_type=["ORIGINAL", "PRIMARY", "M", "ND"], rows=256, columns=256,
        )) == {}

    def test_a_missing_rows_tag_does_not_raise(self) -> None:
        """CSA objects carry no Image Pixel module at all. Reading them with a
        bare getattr raised AttributeError and took the whole scan down."""
        csa_type, image_type = CSA_SPECTROSCOPY
        tags = read_mrs_tags(_dataset(
            sop_class=SIEMENS_CSA_NONIMAGE_SOP_CLASS,
            image_type=image_type, csa_type=csa_type,
        ))
        assert tags["rows"] == 0 and tags["columns"] == 0


# ---------------------------------------------------------------------------
# _classify_mrs_rows: which suffix, and who it may overrule
# ---------------------------------------------------------------------------


def _frame(tags: dict, classifier: str = "", datatype: str = "") -> pd.DataFrame:
    return pd.DataFrame([{
        "_mrs_tags": tags,
        "bids_guess_datatype": datatype,
        "bids_guess_suffix": "",
        "bids_guess_classifier": classifier,
        "bids_guess_confidence": "",
        "modality": "mri",
        # The columns the CONVERTER reads. A row can carry a perfectly good
        # bids_guess_* verdict and still convert to nowhere without these.
        "participant_id": "sub-001",
        "session": "",
        "datatype": "",
        "bids_name": "",
        "bids_path": "",
        "entities": "",
        "issues": "",
    }])


class TestNaming:
    """Classifying a row is not the same as telling the converter where it goes.

    ``bids_guess_*`` is the classifier's opinion; ``datatype`` / ``bids_name`` /
    ``bids_path`` are where the file is actually written, and the converter
    reads the latter. Setting only the former left spectroscopy looking correct
    in the inventory, with the right datatype in the table, and converting to
    nowhere. Nothing errored, because a row with no name is simply not written.
    """

    def test_a_classified_row_gets_a_bids_name(self) -> None:
        df = _frame({"rows": 1, "columns": 1, "localisation": "PRESS"})
        _classify_mrs_rows(df)
        assert df.at[0, "datatype"] == "mrs"
        assert df.at[0, "bids_name"] == "sub-001_svs"
        assert df.at[0, "bids_path"] == "mrs/sub-001_svs.nii.gz"

    def test_the_entities_json_is_written_for_rebuild(self) -> None:
        """``bidsmgr-rebuild`` regenerates the basename from this, so a row
        without it cannot be renamed by an entity edit."""
        import json as _json

        df = _frame({"rows": 1, "columns": 1, "localisation": "PRESS"})
        _classify_mrs_rows(df)
        assert _json.loads(df.at[0, "entities"]) == {"subject": "001"}

    def test_a_session_reaches_the_name(self) -> None:
        df = _frame({"rows": 1, "columns": 1, "localisation": "PRESS"})
        df.at[0, "session"] = "ses-pre"
        _classify_mrs_rows(df)
        assert df.at[0, "bids_name"] == "sub-001_ses-pre_svs"
        assert df.at[0, "bids_path"] == "mrs/sub-001_ses-pre_svs.nii.gz"

    def test_the_suffix_reaches_the_name(self) -> None:
        df = _frame({"rows": 16, "columns": 16, "localisation": "PRESS"})
        _classify_mrs_rows(df)
        assert df.at[0, "bids_name"].endswith("_mrsi")
        assert df.at[0, "bids_path"] == "mrs/sub-001_mrsi.nii.gz"

    def test_no_participant_means_no_name_rather_than_a_bad_one(self) -> None:
        df = _frame({"rows": 1, "columns": 1, "localisation": "PRESS"})
        df.at[0, "participant_id"] = ""
        _classify_mrs_rows(df)
        assert df.at[0, "bids_name"] == ""
        assert df.at[0, "bids_guess_datatype"] == "mrs", (
            "the classification still stands; only the name could not be built"
        )


class TestSuffix:
    """BIDS splits the mrs suffixes on facts the header carries."""

    def test_one_voxel_is_svs(self) -> None:
        df = _frame({"rows": 1, "columns": 1, "localisation": "PRESS"})
        _classify_mrs_rows(df)
        assert df.at[0, "bids_guess_datatype"] == "mrs"
        assert df.at[0, "bids_guess_suffix"] == "svs"
        assert df.at[0, "bids_guess_classifier"] == "dicom_spectroscopy"

    def test_a_grid_of_voxels_is_mrsi(self) -> None:
        df = _frame({"rows": 16, "columns": 16, "localisation": "PRESS"})
        _classify_mrs_rows(df)
        assert df.at[0, "bids_guess_suffix"] == "mrsi"

    def test_explicitly_unlocalised_is_unloc(self) -> None:
        df = _frame({"rows": 1, "columns": 1, "localisation": "NONE"})
        _classify_mrs_rows(df)
        assert df.at[0, "bids_guess_suffix"] == "unloc"

    def test_an_absent_localisation_tag_is_not_unloc(self) -> None:
        """Absence is not a claim. Siemens' CSA objects carry no localisation
        tag at all and are plainly voxel spectroscopy, so reading the gap as
        "unlocalised" would assert something the header never said."""
        df = _frame({"rows": 0, "columns": 0, "localisation": ""})
        _classify_mrs_rows(df)
        assert df.at[0, "bids_guess_suffix"] == "svs"

    def test_a_row_with_no_spectroscopy_tags_is_left_alone(self) -> None:
        df = _frame({})
        _classify_mrs_rows(df)
        assert df.at[0, "bids_guess_datatype"] == ""


class TestPrecedence:
    """Stronger than a name pattern, weaker than dcm2niix."""

    def test_it_overrules_the_regex_layer(self) -> None:
        """The reported case: ``task-xx_..._svs`` matched ``task-`` and was
        filed as func/bold, then dropped for carrying no pixel data."""
        df = _frame(
            {"rows": 0, "columns": 0, "localisation": ""},
            classifier="sequence_dict", datatype="func",
        )
        _classify_mrs_rows(df)
        assert df.at[0, "bids_guess_datatype"] == "mrs"

    def test_notes_about_the_old_verdict_are_dropped(self) -> None:
        """"Required entity 'task' missing" is a fact about ``sbref``, which
        needs one, not about ``svs``, which does not. Left in place it reads
        as an unfixable problem with a row that is now correct."""
        df = _frame(
            {"rows": 0, "columns": 0, "localisation": ""},
            classifier="sequence_dict", datatype="func",
        )
        df.at[0, "issues"] = "entity.required: Required entity 'task' missing"
        _classify_mrs_rows(df)
        assert df.at[0, "issues"] == ""

    def test_notes_about_the_data_survive(self) -> None:
        """A repeat scan is still a repeat scan whatever it is classified as,
        and a user exclusion is still the user's decision."""
        df = _frame(
            {"rows": 0, "columns": 0, "localisation": ""},
            classifier="sequence_dict", datatype="func",
        )
        df.at[0, "issues"] = (
            "entity.required: Required entity 'task' missing "
            "| suspected_abort: same SeriesDescription as a later companion"
        )
        _classify_mrs_rows(df)
        assert "suspected_abort" in df.at[0, "issues"]
        assert "entity.required" not in df.at[0, "issues"]

    def test_it_overrules_a_rerouted_regex_verdict_too(self) -> None:
        assert _mrs_may_overrule("sequence_dict+b0_reroute")

    def test_it_never_overrules_dcm2niix(self) -> None:
        df = _frame(
            {"rows": 1, "columns": 1, "localisation": "PRESS"},
            classifier="dcm2niix_bidsguess", datatype="anat",
        )
        _classify_mrs_rows(df)
        assert df.at[0, "bids_guess_datatype"] == "anat"
        assert not _mrs_may_overrule("dcm2niix_bidsguess")
        assert not _mrs_may_overrule("dcm2niix_bidsguess+b0_reroute")

    def test_it_fills_a_blank(self) -> None:
        assert _mrs_may_overrule("")


# ---------------------------------------------------------------------------
# End to end: the header survives the scan and keeps the row convertible
# ---------------------------------------------------------------------------


def test_spectroscopy_reaches_the_inventory_as_mrs(tmp_path: Path) -> None:
    """Through ``_read_one`` and the per-series aggregation, not around them."""
    from bidsmgr.inventory.mri_dicom import _read_one

    path = tmp_path / "svs.dcm"
    ds = _dataset(
        sop_class=MR_SPECTROSCOPY_SOP_CLASS,
        image_type=["ORIGINAL", "PRIMARY", "SPECTROSCOPY", "NONE"],
        rows=1, columns=1, localisation="PRESS", nucleus="1H",
    )
    ds.PatientID = "P1"
    ds.PatientName = "Doe^Jane"
    ds.SeriesDescription = "svs_se"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.save_as(str(path), enforce_file_format=True)

    res = _read_one(str(path), tmp_path)
    assert res is not None
    assert res["mrs"], "the per-file read must carry the spectroscopy facts"

    df = _frame(res["mrs"])
    _classify_mrs_rows(df)
    assert df.at[0, "bids_guess_datatype"] == "mrs"
    assert df.at[0, "bids_guess_suffix"] == "svs"


def test_a_classified_row_survives_the_nonimage_cull() -> None:
    """The point of classifying it at all: spectroscopy carries no Image Pixel
    module, so without the datatype it is dropped as a non-image series."""
    from bidsmgr.cli.scan import _flag_nonimage_rows, _is_spectroscopy_row

    df = pd.DataFrame([{
        "_has_pixel_data": False,
        "bids_guess_datatype": "mrs",
        "bids_guess_suffix": "svs",
        "bids_guess_skip": False,
        "include": 1,
        "issues": "",
        "sequence": "svs_se",
        "BIDS_name": "sub-001",
    }])
    assert _is_spectroscopy_row(df, 0)
    _flag_nonimage_rows(df)
    assert df.at[0, "include"] == 1
    # ``not`` rather than ``is False``: pandas hands back a numpy bool.
    assert not df.at[0, "bids_guess_skip"]
