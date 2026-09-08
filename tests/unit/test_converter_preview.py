"""What the conversion answers by itself, and why the form should say so.

Two complaints this addresses, both real. A form that omits a required field
because a converter fills it looks like a form that forgot it. And the list of
"what a converter fills" was measured once, on one tree, with one scanner, so a
field the user's scanner supplies but ours did not was asked for AND reported as
already filled in, in the same form.
"""

from __future__ import annotations

from types import SimpleNamespace

import json

import pandas as pd

from bidsmgr.metadata.converter_preview import (
    merge_previews,
    preview_from_inventory,
    preview_from_probe,
)
from bidsmgr.metadata.template_plan import sidecar_section
from bidsmgr.recording_meta import VARIES


def _eeg_rows(*derived: dict, task: str = "rest") -> pd.DataFrame:
    """Rows shaped as the scanner writes them: one JSON blob per recording of
    exactly what mne-bids will derive from it."""
    return pd.DataFrame([
        {
            "include": "1", "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
            "task": task, "_derived_fields": json.dumps(d),
        }
        for d in derived
    ])


def test_the_scan_alone_previews_a_recording() -> None:
    """No conversion has to run: the scanner asked the recording the same
    questions mne-bids will ask."""
    derived = {"SamplingFrequency": 500.0, "EEGChannelCount": 64}
    preview = preview_from_inventory(_eeg_rows(derived, derived))
    assert preview["eeg/eeg"] == {
        "EEGChannelCount": 64,
        "SamplingFrequency": 500.0,
        "TaskName": "rest",
    }


def test_disagreement_is_reported_as_varies_not_as_a_value() -> None:
    """One recording's value is not a statement about the kind. VARIES says the
    answer lives per recording, which is what that sentinel is for."""
    preview = preview_from_inventory(_eeg_rows(
        {"SamplingFrequency": 500.0, "EEGChannelCount": 64},
        {"SamplingFrequency": 500.0, "EEGChannelCount": 32},
    ))
    assert preview["eeg/eeg"]["EEGChannelCount"] == VARIES
    assert preview["eeg/eeg"]["SamplingFrequency"] == 500.0


def test_a_field_only_some_files_answer_reads_as_varying() -> None:
    """Not asked, and not claimed as one answer either.

    Requiring every file to answer before saying the conversion does was the
    first attempt, and it was too strict for real data: seventy-five functional
    runs from several studies agree on fourteen fields and no more, so the form
    asked for the repetition time and thirty others that dcm2niix reads out of
    the header for almost all of them.

    VARIES is the honest answer, and the block is editable, so a value stated
    there covers the recordings whose header lacked one.
    """
    preview = preview_from_inventory(_eeg_rows(
        {"SamplingFrequency": 500.0, "Manufacturer": "BioSemi"},
        {"SamplingFrequency": 500.0},
    ))
    assert preview["eeg/eeg"]["SamplingFrequency"] == 500.0
    assert preview["eeg/eeg"]["Manufacturer"] == VARIES


def test_excluded_rows_do_not_contribute() -> None:
    df = _eeg_rows(
        {"EEGChannelCount": 64}, {"EEGChannelCount": 32},
    )
    df.loc[1, "include"] = "0"
    assert preview_from_inventory(df)["eeg/eeg"]["EEGChannelCount"] == 64


def test_a_placeholder_is_not_an_answer() -> None:
    """BIDS demands the key; mne-bids writes n/a when it does not know. Counting
    that as answered is what hid EEGReference and EEGGround from the form."""
    df = pd.DataFrame([{
        "include": "1", "series_uid": "1.2.3",
        "proposed_datatype": "eeg", "bids_guess_suffix": "eeg",
    }])
    stats = {"1.2.3": SimpleNamespace(sidecar_fields={
        "EEGReference": "n/a", "SamplingFrequency": 500.0,
    })}
    preview = preview_from_probe(df, stats)["eeg/eeg"]
    assert "SamplingFrequency" in preview
    assert "EEGReference" not in preview


def test_a_probe_previews_what_dcm2niix_wrote() -> None:
    df = pd.DataFrame([{
        "include": "1", "series_uid": "1.2.3",
        "proposed_datatype": "anat", "bids_guess_suffix": "T1w",
    }])
    stats = {"1.2.3": SimpleNamespace(sidecar_fields={
        "EchoTime": 0.003, "FlipAngle": 8, "ConversionSoftware": "dcm2niix",
    })}
    preview = preview_from_probe(df, stats)["anat/T1w"]
    assert preview == {"EchoTime": 0.003, "FlipAngle": 8}, (
        "the converter's own bookkeeping keys are not BIDS and must not appear"
    )


def test_a_field_the_datatype_does_not_take_is_dropped() -> None:
    """dcm2niix writes keys BIDS does not declare for the file it wrote."""
    df = pd.DataFrame([{
        "include": "1", "series_uid": "1.2.3",
        "proposed_datatype": "anat", "bids_guess_suffix": "T1w",
    }])
    stats = {"1.2.3": SimpleNamespace(sidecar_fields={
        "EchoTime": 0.003, "TracerName": "FDG",
    })}
    assert "TracerName" not in preview_from_probe(df, stats)["anat/T1w"]


def test_the_template_stops_asking_for_what_the_conversion_answers() -> None:
    """The point of the whole thing: no field is both asked and reported filled.

    ``AcquisitionDuration`` is the real case. dcm2niix writes it on the scanner
    this was measured against and not on the one the built-in list came from, so
    before this it was asked for in the same form that reported it filled in.
    """
    assert "AcquisitionDuration" in {
        f.name for f in sidecar_section("anat", "T1w").fields
    }
    answered = {"AcquisitionDuration": 300.0}
    asked = {f.name for f in sidecar_section("anat", "T1w", answered=answered).fields}
    assert not (asked & set(answered))


def test_a_blank_preview_value_still_asks() -> None:
    """A converter that wrote the key empty has not answered it."""
    asked = {
        f.name
        for f in sidecar_section("anat", "T1w", answered={"AcquisitionDuration": ""}).fields
    }
    assert "AcquisitionDuration" in asked


def test_later_sources_win_when_merged() -> None:
    """The probe saw the real conversion; the inventory only inferred it."""
    merged = merge_previews(
        {"eeg/eeg": {"SamplingFrequency": 500.0, "TaskName": "rest"}},
        {"eeg/eeg": {"SamplingFrequency": 1000.0}},
    )
    assert merged["eeg/eeg"] == {"SamplingFrequency": 1000.0, "TaskName": "rest"}


def test_each_recording_gets_its_own_measurement() -> None:
    """What EVERY file of a kind answers is not the same question as what THIS
    file answers, and the per-file form asks the second one.

    Only the EEG scanner used to record it, so an MRI row could say nothing
    about what its conversion answers and its form asked for the repetition
    time dcm2niix reads straight out of the header.
    """
    from bidsmgr.metadata.converter_preview import preview_by_row

    df = pd.DataFrame([
        {"include": "1", "proposed_datatype": "func", "bids_guess_suffix": "bold",
         "series_uid": "1.2.3",
         "_derived_fields": json.dumps({"RepetitionTime": 2.5, "EchoTime": 0.03})},
        {"include": "1", "proposed_datatype": "func", "bids_guess_suffix": "bold",
         "series_uid": "1.2.4",
         "_derived_fields": json.dumps({"RepetitionTime": 3.0})},
    ])
    by_row = preview_by_row(df)
    assert by_row["1.2.3"] == {"RepetitionTime": 2.5, "EchoTime": 0.03}
    assert by_row["1.2.4"] == {"RepetitionTime": 3.0}


def test_a_per_row_measurement_drops_what_the_file_cannot_carry() -> None:
    from bidsmgr.metadata.converter_preview import preview_by_row

    df = pd.DataFrame([{
        "include": "1", "proposed_datatype": "anat", "bids_guess_suffix": "T1w",
        "series_uid": "1.2.3",
        "_derived_fields": json.dumps({
            "EchoTime": 0.03, "TracerName": "FDG", "ConversionSoftware": "dcm2niix",
        }),
    }])
    assert preview_by_row(df)["1.2.3"] == {"EchoTime": 0.03}
