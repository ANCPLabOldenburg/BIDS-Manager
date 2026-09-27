"""One fieldmap acquisition is one row, and two are two.

A gradient-echo fieldmap reaches the scanner as two DICOM series, magnitude
and phase, which ``_collapse_fieldmap_rows`` merges into the single inventory
row that produces ``magnitude1`` + ``magnitude2`` + ``phasediff`` after
conversion. Nothing here tested that, which is how it came to merge every
fieldmap in a session into one row on any dataset whose DICOM omits
``AcquisitionTime``.

The defect that prompted these: ``raw_data/MRI/SIna_data/OL_4925``, a Siemens
XA study with two fieldmaps (series 6+7 and 12+13) all sharing the
SeriesDescription ``fmap``. The old key bucketed on ``acq_time[:4]``, the
clock minute, and XA writes ``AcquisitionDateTime`` instead of
``AcquisitionTime``, so every row bucketed on ``""``. All four series became
one row, and conversion wrote six images of which three carried dcm2niix's
own collision suffixes (``_e1a``, ``_e2a``, ``_e2_pha``) instead of BIDS
names, one of them calling a phase image ``magnitude1``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bidsmgr.inventory.mri_dicom import (
    _collapse_fieldmap_rows,
    acquisition_time,
)


def _row(uid: str, image_type: str, acq_time: str = "", **over):
    """One pre-collapse fieldmap row, with only the columns the merge reads."""
    row = {
        "subject": "OL_4925",
        "participant_id": "sub-001",
        "session": "",
        "source_folder": "OL_4925",
        "include": 1,
        "sequence": "fmap",
        "series_uid": uid,
        "image_type": image_type,
        "acq_time": acq_time,
        "modality": "fmap",
        "sequence_kind": "fmap",
        "n_files": 1,
        "study_instance_uid": "1.2.3",
        "study_date": "20260917",
        "study_time": "090341",
        "_source_dir": "/raw/OL_4925",
        "GivenName": "",
        "FamilyName": "",
        "PatientID": "OL_4925",
        "PatientSex": "M",
        "PatientAge": "030Y",
        "StudyDescription": "study",
    }
    row.update(over)
    return row


def _collapse(rows):
    return _collapse_fieldmap_rows(pd.DataFrame(rows)).reset_index(drop=True)


# -- the reported defect ------------------------------------------------


def test_two_acquisitions_stay_two_rows_without_acquisition_time():
    """The regression. Four series, no clock, two fieldmaps."""
    out = _collapse([
        _row("uid.6", "M"),
        _row("uid.7", "P"),
        _row("uid.12", "M"),
        _row("uid.13", "P"),
    ])
    assert len(out) == 2
    for _, r in out.iterrows():
        assert r["series_uid"].count("|") == 1, "one magnitude and one phase"
        assert sorted(r["image_type"]) == ["M", "P"]


def test_the_two_rows_keep_their_own_series():
    """Not just two rows: the RIGHT two. A merge that split 6+12 and 7+13
    would also produce two rows and would be silently wrong."""
    out = _collapse([
        _row("uid.6", "M"),
        _row("uid.7", "P"),
        _row("uid.12", "M"),
        _row("uid.13", "P"),
    ])
    pairs = {frozenset(r["series_uid"].split("|")) for _, r in out.iterrows()}
    assert pairs == {frozenset({"uid.6", "uid.7"}),
                     frozenset({"uid.12", "uid.13"})}


def test_two_acquisitions_are_numbered_for_the_run_entity():
    out = _collapse([
        _row("uid.6", "M", "093131.650000"),
        _row("uid.7", "P", "093132.052500"),
        _row("uid.12", "M", "095453.052500"),
        _row("uid.13", "P", "095453.455000"),
    ])
    assert sorted(str(v) for v in out["rep"]) == ["1", "2"]
    # Chronological: the earlier acquisition is run 1.
    first = out[out["rep"].astype(str) == "1"].iloc[0]
    assert "uid.6" in first["series_uid"]


# -- the behaviour that was already right -------------------------------


def test_one_acquisition_is_one_row():
    out = _collapse([_row("uid.6", "M"), _row("uid.7", "P")])
    assert len(out) == 1
    assert out.iloc[0]["series_uid"] == "uid.6|uid.7"
    assert out.iloc[0]["n_files"] == 2
    assert out.iloc[0]["rep"] == "", "a lone fieldmap gets no run number"


def test_a_pair_split_across_a_clock_minute_still_merges():
    """The other failure of the old key: ``093159`` and ``093200`` bucketed
    apart on ``[:4]`` and the pair was never merged."""
    out = _collapse([
        _row("uid.6", "M", "093159.900000"),
        _row("uid.7", "P", "093200.100000"),
    ])
    assert len(out) == 1


def test_different_sequences_never_merge():
    """``ses-pre_run-01_fmap`` and ``ses-pre_run-02_fmap`` are already
    distinct by SeriesDescription; the grouping must keep them so."""
    out = _collapse([
        _row("uid.2", "M", sequence="ses-pre_run-01_fmap"),
        _row("uid.3", "P", sequence="ses-pre_run-01_fmap"),
        _row("uid.7", "M", sequence="ses-pre_run-02_fmap"),
        _row("uid.8", "P", sequence="ses-pre_run-02_fmap"),
    ])
    assert len(out) == 2


def test_different_subjects_never_merge():
    out = _collapse([
        _row("uid.6", "M", participant_id="sub-001"),
        _row("uid.7", "P", participant_id="sub-001"),
        _row("uid.6b", "M", participant_id="sub-002"),
        _row("uid.7b", "P", participant_id="sub-002"),
    ])
    assert len(out) == 2
    assert set(out["participant_id"]) == {"sub-001", "sub-002"}


def test_non_fieldmap_rows_are_untouched():
    df = pd.DataFrame([
        _row("uid.5", "M", sequence="T1w", modality="anat"),
        _row("uid.6", "M"),
        _row("uid.7", "P"),
    ])
    out = _collapse_fieldmap_rows(df)
    anat = out[out["modality"] == "anat"]
    assert len(anat) == 1 and anat.iloc[0]["series_uid"] == "uid.5"


def test_an_unknown_image_type_merges_rather_than_splits():
    """A vendor that does not say what the image is gets the old behaviour.
    Splitting on a value we cannot read would invent acquisitions."""
    out = _collapse([_row("uid.6", ""), _row("uid.7", "")])
    assert len(out) == 1


def test_no_fieldmaps_at_all_is_a_no_op():
    df = pd.DataFrame([_row("uid.5", "M", sequence="T1w", modality="anat")])
    out = _collapse_fieldmap_rows(df)
    assert len(out) == 1


# -- the time source ----------------------------------------------------


class _Fake:
    """Stands in for a pydicom Dataset: attribute access is all that is read."""

    def __init__(self, **tags):
        for k, v in tags.items():
            setattr(self, k, v)


def test_acquisition_time_prefers_the_acquisition_tag():
    ds = _Fake(AcquisitionTime="093131.650000",
               AcquisitionDateTime="20260917095959.000000",
               SeriesTime="101010.000000")
    assert acquisition_time(ds) == "093131.650000"


def test_acquisition_time_falls_back_to_the_datetime_tag():
    """Siemens XA. The time half only, so the column means one thing."""
    ds = _Fake(AcquisitionDateTime="20260917093131.650000")
    assert acquisition_time(ds) == "093131.650000"


def test_acquisition_time_falls_back_to_series_then_content_time():
    assert acquisition_time(_Fake(SeriesTime="093223.050000")) == "093223.050000"
    assert acquisition_time(_Fake(ContentTime="093223.623000")) == "093223.623000"


@pytest.mark.parametrize("value", ["", "not-a-datetime", "2026091709"])
def test_acquisition_time_rejects_a_malformed_datetime(value):
    assert acquisition_time(_Fake(AcquisitionDateTime=value)) == ""


def test_acquisition_time_is_empty_when_the_dicom_says_nothing():
    assert acquisition_time(_Fake()) == ""
