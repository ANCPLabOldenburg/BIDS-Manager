"""``*_scans.tsv`` describes every recording, not only the images.

BIDS is explicit: "Each neural recording file SHOULD be described by exactly
one row." The generator globbed ``*.nii*``, which had three consequences on any
dataset holding EEG, MEG, iEEG or ECAT PET:

* a subject with no NIfTI at all produced NO scans table;
* a mixed session produced one that listed the images and silently omitted the
  recordings beside them;
* and because mne-bids WRITES this table during conversion, with a real
  ``acq_time`` read from the recording's ``meas_date``, regenerating it from
  the NIfTIs DELETED those rows and the only acquisition times the dataset had
  for its recordings.

The third is the one that made it look like a rename bug as well. There was
nothing wrong with the rename: it had no rows to move because the table it
would have moved them in had been emptied of everything but the images.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.metadata.engine import run_metadata


def _write(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content), encoding="utf-8")
    elif isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(str(content), encoding="utf-8")


@pytest.fixture
def mixed(tmp_path: Path) -> Path:
    """One session carrying MRI, EEG and MEG, plus an EEG-only subject."""
    root = tmp_path / "ds"
    ses = root / "sub-01" / "ses-01"

    _write(ses / "anat" / "sub-01_ses-01_T1w.nii.gz", b"\0" * 64)
    _write(ses / "anat" / "sub-01_ses-01_T1w.json", {"EchoTime": 0.03})
    _write(ses / "func" / "sub-01_ses-01_task-rest_bold.nii.gz", b"\0" * 64)
    _write(ses / "func" / "sub-01_ses-01_task-rest_bold.json", {"RepetitionTime": 2.0})

    # BrainVision is three files and ONE recording: the header gets the row.
    _write(ses / "eeg" / "sub-01_ses-01_task-rest_eeg.vhdr", "hdr")
    _write(ses / "eeg" / "sub-01_ses-01_task-rest_eeg.eeg", b"\0" * 8)
    _write(ses / "eeg" / "sub-01_ses-01_task-rest_eeg.vmrk", "mrk")
    _write(ses / "eeg" / "sub-01_ses-01_task-rest_eeg.json", {"TaskName": "rest"})
    _write(ses / "eeg" / "sub-01_ses-01_task-rest_channels.tsv", "name\ttype\nCz\tEEG\n")
    _write(ses / "eeg" / "sub-01_ses-01_task-rest_events.tsv", "onset\tduration\n1\t0\n")

    _write(ses / "meg" / "sub-01_ses-01_task-rest_meg.fif", b"\0" * 8)
    _write(ses / "meg" / "sub-01_ses-01_task-rest_meg.json", {"TaskName": "rest"})
    # CTF: a recording that is a DIRECTORY. One row, not one per internal file.
    _write(ses / "meg" / "sub-01_ses-01_task-other_meg.ds" / "x.meg4", b"\0")
    # MEG system files mne-bids writes beside the data. Not recordings.
    _write(ses / "meg" / "sub-01_ses-01_acq-calibration_meg.dat", b"\0")
    _write(ses / "meg" / "sub-01_ses-01_acq-crosstalk_meg.fif", b"\0")

    _write(root / "sub-02" / "eeg" / "sub-02_task-rest_eeg.edf", b"\0" * 8)
    _write(root / "sub-02" / "eeg" / "sub-02_task-rest_eeg.json", {"TaskName": "rest"})

    _write(root / "dataset_description.json",
           {"Name": "mixed", "BIDSVersion": "1.10.0", "DatasetType": "raw"})
    return root


def _rows(table: Path) -> list[str]:
    frame = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
    return list(frame["filename"])


def _table(root: Path, rel: str) -> Path:
    return root / rel


# ---------------------------------------------------------------------------
# Every modality gets a row
# ---------------------------------------------------------------------------


def test_the_eeg_recording_is_listed(mixed: Path) -> None:
    run_metadata(mixed)
    rows = _rows(_table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv"))
    assert "eeg/sub-01_ses-01_task-rest_eeg.vhdr" in rows


def test_the_meg_recording_is_listed(mixed: Path) -> None:
    run_metadata(mixed)
    rows = _rows(_table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv"))
    assert "meg/sub-01_ses-01_task-rest_meg.fif" in rows


def test_a_folder_recording_is_one_row(mixed: Path) -> None:
    """CTF ``.ds`` is a directory. One row for it, and none for its contents."""
    run_metadata(mixed)
    rows = _rows(_table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv"))
    assert "meg/sub-01_ses-01_task-other_meg.ds" in rows
    assert not [r for r in rows if ".ds/" in r]


def test_the_images_are_still_listed(mixed: Path) -> None:
    """The fix must not trade one modality for another."""
    run_metadata(mixed)
    rows = _rows(_table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv"))
    assert "anat/sub-01_ses-01_T1w.nii.gz" in rows
    assert "func/sub-01_ses-01_task-rest_bold.nii.gz" in rows


def test_a_subject_with_no_images_still_gets_a_table(mixed: Path) -> None:
    """It produced nothing at all before: no NIfTI, no rows, no file."""
    run_metadata(mixed)
    table = _table(mixed, "sub-02/sub-02_scans.tsv")
    assert table.is_file(), "an EEG-only subject needs a scans table too"
    assert _rows(table) == ["eeg/sub-02_task-rest_eeg.edf"]


# ---------------------------------------------------------------------------
# Companions do NOT get a row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("companion", [
    "eeg/sub-01_ses-01_task-rest_eeg.eeg",      # BrainVision binary
    "eeg/sub-01_ses-01_task-rest_eeg.vmrk",     # BrainVision markers
    "eeg/sub-01_ses-01_task-rest_eeg.json",
    "eeg/sub-01_ses-01_task-rest_channels.tsv",
    "eeg/sub-01_ses-01_task-rest_events.tsv",
    "anat/sub-01_ses-01_T1w.json",
    "meg/sub-01_ses-01_acq-calibration_meg.dat",
    "meg/sub-01_ses-01_acq-crosstalk_meg.fif",
])
def test_companions_get_no_row(mixed: Path, companion: str) -> None:
    """A recording spread over several files is still one recording."""
    run_metadata(mixed)
    rows = _rows(_table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv"))
    assert companion not in rows


def test_the_table_does_not_list_itself(mixed: Path) -> None:
    run_metadata(mixed)
    rows = _rows(_table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv"))
    assert not [r for r in rows if r.endswith("_scans.tsv")]


# ---------------------------------------------------------------------------
# What mne-bids measured is not thrown away
# ---------------------------------------------------------------------------


def test_an_existing_acq_time_survives(mixed: Path) -> None:
    """mne-bids reads meas_date off the recording and writes it here. Nothing
    else in the dataset holds it, so regenerating the table destroyed it."""
    table = _table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv")
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(
        "filename\tacq_time\n"
        "eeg/sub-01_ses-01_task-rest_eeg.vhdr\t2024-03-01T09:15:00.000000Z\n",
        encoding="utf-8",
    )
    run_metadata(mixed)
    frame = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
    kept = frame.loc[
        frame["filename"] == "eeg/sub-01_ses-01_task-rest_eeg.vhdr", "acq_time"
    ]
    assert list(kept) == ["2024-03-01T09:15:00.000000Z"]


def test_extra_columns_survive(mixed: Path) -> None:
    """mne-bids adds ``source`` when asked, and a user may add their own."""
    table = _table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv")
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(
        "filename\tacq_time\tsource\n"
        "eeg/sub-01_ses-01_task-rest_eeg.vhdr\tn/a\t/raw/subj1.vhdr\n",
        encoding="utf-8",
    )
    run_metadata(mixed)
    frame = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
    assert "source" in frame.columns
    row = frame.loc[frame["filename"] == "eeg/sub-01_ses-01_task-rest_eeg.vhdr"]
    assert list(row["source"]) == ["/raw/subj1.vhdr"]


def test_a_row_whose_file_is_gone_is_dropped(mixed: Path) -> None:
    """A stale filename is SCANS_FILENAME_NOT_MATCH_DATASET, an error. After a
    rename the old name is exactly that, so merging must not keep it."""
    table = _table(mixed, "sub-01/ses-01/sub-01_ses-01_scans.tsv")
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(
        "filename\tacq_time\n"
        "eeg/sub-01_ses-01_task-GONE_eeg.vhdr\t2024-01-01T00:00:00.000000Z\n",
        encoding="utf-8",
    )
    run_metadata(mixed)
    rows = _rows(table)
    assert "eeg/sub-01_ses-01_task-GONE_eeg.vhdr" not in rows
    assert "eeg/sub-01_ses-01_task-rest_eeg.vhdr" in rows


# ---------------------------------------------------------------------------
# The validator agrees
# ---------------------------------------------------------------------------


def test_the_result_raises_no_scans_findings(mixed: Path) -> None:
    from bidsmgr.editor.validator import validate

    run_metadata(mixed)
    report = validate(mixed)
    found = [
        i.rule_id for v in report.files for i in (v.issues or [])
        if "SCANS" in i.rule_id and not getattr(i, "mirrored", False)
    ] + [
        i.rule_id for i in (report.dataset_issues or []) if "SCANS" in i.rule_id
    ]
    assert not found, found


# ---------------------------------------------------------------------------
# And the rename, which was the second half of the same bug
# ---------------------------------------------------------------------------


def test_renaming_a_subject_carries_its_recordings(mixed: Path) -> None:
    """The rename was never modality-specific. It had nothing to move because
    the table held no rows for the recordings."""
    from bidsmgr.editor.rename import apply_rename, plan_rename

    run_metadata(mixed)
    plan = plan_rename(mixed, "sub", "01", "42")
    _touched, errors = apply_rename(mixed, plan)
    assert not errors, errors

    table = _table(mixed, "sub-42/ses-01/sub-42_ses-01_scans.tsv")
    assert table.is_file()
    rows = _rows(table)
    assert "eeg/sub-42_ses-01_task-rest_eeg.vhdr" in rows
    assert "meg/sub-42_ses-01_task-rest_meg.fif" in rows
    assert "anat/sub-42_ses-01_T1w.nii.gz" in rows
    assert not [r for r in rows if "sub-01" in r], "no stale names"
